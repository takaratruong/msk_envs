"use strict";
(() => {
  const $=id=>document.getElementById(id), keys=new Set(), resets=[];
  const controlKeys=new Set(["KeyW","KeyA","KeyS","KeyD","KeyQ","KeyE","KeyV","ArrowUp","ArrowDown","ArrowLeft","ArrowRight","KeyF","Space","Escape","Enter","NumpadEnter"]);
  const clamp=(x,a=-1,b=1)=>Math.max(a,Math.min(b,x));
  const wrap=x=>Math.atan2(Math.sin(x),Math.cos(x));
  const dead=x=>Number.isFinite(x)&&Math.abs(x)>=.15?Math.sign(x)*(Math.abs(x)-.15)/.85:0;
  const fmt=(x,d=1)=>Number.isFinite(x)?x.toFixed(d):"—";
  const INTERACTION_ACTIONS=Object.freeze(["open","close","grasp","place"]);
  const ACTION_LABELS=Object.freeze({open:"Open",close:"Close",grasp:"Grasp soda",place:"Place in blue tray"});
  const MISSION_OBJECTIVES=Object.freeze({find:"Find the soda",carrying:"Carry the soda to the blue tray",placing:"Place soda in the blue tray",complete:"Soda delivered",failed:"Soda task stopped"});
  const INPUT_PERIOD_MS=20,VIEW_PERIOD_MS=1000/60;
  const MOUSE_RADIANS_PER_PIXEL=.00075,MOUSE_SENSITIVITY_KEY="g1.roomGame.mouseSensitivity.v1";
  const MOUSE_MAX_EVENT_PIXELS=100,MOUSE_MAX_RADIANS_PER_SECOND=5*Math.PI/3;
  let mouseSensitivity=1;
  let mouseAngleRemaining=0,skipCapturedMouse=true;
  try{const saved=Number(localStorage.getItem(MOUSE_SENSITIVITY_KEY));if(Number.isFinite(saved)&&saved>=.25&&saved<=2)mouseSensitivity=saved;}catch{}
  function renderMouseSensitivity(){
    $("mouse-sensitivity").value=String(mouseSensitivity);
    $("mouse-sensitivity-value").textContent=`${mouseSensitivity.toFixed(2)}×`;
  }
  $("mouse-sensitivity").addEventListener("input",e=>{
    const value=Number(e.target.value);if(!Number.isFinite(value)||value<.25||value>2)return;
    mouseSensitivity=value;renderMouseSensitivity();try{localStorage.setItem(MOUSE_SENSITIVITY_KEY,String(value));}catch{}
  });
  $("mouse-sensitivity").addEventListener("keydown",e=>{if(e.code.startsWith("Arrow"))e.stopPropagation();});
  renderMouseSensitivity();
  function applyMouseDelta(dx,dy){
    if(!Number.isFinite(dx)||!Number.isFinite(dy)||Math.abs(dx)>MOUSE_MAX_EVENT_PIXELS||Math.abs(dy)>MOUSE_MAX_EVENT_PIXELS)return;
    const gain=MOUSE_RADIANS_PER_PIXEL*mouseSensitivity,yawDelta=-dx*gain;
    const pitchDelta=clamp(look.pitch-dy*gain,-1.15,.60)-look.pitch,angle=Math.hypot(yawDelta,pitchDelta);
    if(!angle||mouseAngleRemaining<=0)return;
    // Apply ordinary deltas immediately. Drop remote recenter bursts and any
    // excess over this animation tick's budget; never replay a backlog.
    const scale=Math.min(1,mouseAngleRemaining/angle);mouseAngleRemaining-=angle*scale;
    setLook(look.heading_world+yawDelta*scale,look.pitch+pitchDelta*scale);
  }
  let token=null,client=null,ticket=null,seq=0,enabled=false,enabling=false;
  let enableGeneration=0,frameGeneration=0,inputFlight=false,stateFlight=false,viewFlight=false;
  let lastState=0,lastStatus={},activeViewSession=null,displayed=null,blobURL=null;
  let look={yaw:0,pitch:-.15,revision:0,heading_world:0,mode:"third_person"},move=[0,0],turn=0,padAxes=[0,0,0],lookStick=[0,0];
  let nativeHeading=null,nativeHeadingTime=-Infinity,mouseCaptured=false,captureUnavailable=false;
  let fPulse=null,fHeld=false,padF=false,padStop=false,touchF=false,mustRelease=false,padBlocked=true;
  let padName="",lastPadTime=performance.now(),drag=null;
  async function json(url,body,keepalive=false){
    const opts={cache:"no-store"};
    if(body!==undefined)Object.assign(opts,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body),keepalive});
    const r=await fetch(url,opts),v=await r.json();if(!r.ok)throw new Error(v.error||`HTTP ${r.status}`);return v;
  }
  function age(frame=displayed){return frame?frame.age_s+(performance.now()-frame.request_started)/1000:Infinity;}
  function frameCurrent(){return displayed&&displayed.view_session===activeViewSession&&displayed.view_revision===look.revision&&displayed.camera.mode===look.mode&&Math.abs(wrap(displayed.camera.desired_heading_world_rad-look.heading_world))<1e-5&&age()<=.30;}
  function observeHeading(value,at){
    if(Number.isFinite(value)&&Math.abs(value)<=Math.PI+1e-6&&Number.isFinite(at)&&at>=nativeHeadingTime){nativeHeading=wrap(value);nativeHeadingTime=at;}
  }
  function bodyTurnHeld(){return enabled&&lastStatus.mode&&!["walking","idle"].includes(lastStatus.mode);}
  function renderLook(){
    const locked=document.pointerLockElement===$("viewport");
    $("view-toggle").textContent=look.mode==="third_person"?"View: Third person · V":"View: Head · V";
    $("view-toggle").disabled=!enabled;
    $("mouse-lock-state").textContent=locked?"Mouse locked · Escape releases":captureUnavailable?"Keyboard look · WASD":"Keyboard look · WASD";
    $("mouse-lock-state").classList.toggle("offline",captureUnavailable&&!locked);
    $("capture-mouse").textContent=locked?"Mouse locked · Escape releases":"Use mouse look";
    $("look-status").textContent=bodyTurnHeld()?"Body turn held during interaction":locked?"Mouse look on · Esc releases · V changes view":captureUnavailable?"Mouse lock failed. Click “Use mouse look” to retry.":"WASD looks · Arrow keys walk · F interacts · V changes view";
    $("viewport").classList.toggle("mouse-captured",locked);
    $("viewport").dataset.mouseCaptured=String(locked);
  }
  function candidate(){return displayed?.selection||null;}
  function candidateFeedback(c){
    const f=lastStatus.interaction_feedback;
    if(!c||!f||f.id!==c.id||f.action!==c.action||typeof f.message!=="string"||!Number.isSafeInteger(f.attempt_id)||f.attempt_id<0||!["planning","approach","align","failed","cancelled","interacting","done"].includes(f.state))return null;
    return f;
  }
  function renderMission(){
    const m=lastStatus.mission,hud=$("mission"),detail=$("mission-detail");
    const valid=m&&typeof m.stage==="string"&&Object.hasOwn(MISSION_OBJECTIVES,m.stage)&&(m.stage!=="complete"||m.completed===true);
    hud.hidden=!valid;detail.hidden=true;
    if(!valid){delete hud.dataset.stage;delete hud.dataset.completed;$("mission-objective").textContent="";detail.textContent="";return;}
    const objective=typeof m.objective==="string"&&m.objective.trim()?m.objective:MISSION_OBJECTIVES[m.stage];
    $("mission-objective").textContent=objective;
    hud.dataset.stage=m.stage;hud.dataset.completed=String(m.stage==="complete"&&m.completed===true);
    hud.classList.toggle("complete",m.stage==="complete");hud.classList.toggle("failed",m.stage==="failed");
    if(m.stage==="failed"&&typeof m.message==="string"&&m.message&&m.message!==objective){detail.textContent=m.message;detail.hidden=false;}
  }
  function renderCandidate(){
    const c=candidate(),current=frameCurrent(),feedback=candidateFeedback(c);
    const objectAction=c?.action==="grasp"||c?.action==="place",label=c?ACTION_LABELS[c.action]:lastStatus.mission?"Interact":"Open";
    const feedbackLabels={planning:"Finding route…",approach:"Walking to target",align:"Lining up",failed:"Interaction stopped",cancelled:"Cancelled",interacting:({open:"Opening",close:"Closing",grasp:"Grasping soda",place:"Placing soda"})[c?.action]||"Interacting",done:"Finished"};
    $("candidate-name").textContent=c?.name||c?.id||(lastStatus.mission?"Aim at something to interact":"Aim at a drawer handle");
    $("candidate-name").hidden=objectAction;$("candidate-separator").hidden=objectAction;
    $("candidate-reason").textContent=feedback?(feedback.message||feedbackLabels[feedback.state]):!displayed?"Waiting for the camera":!current?(displayed.view_revision!==look.revision?"Updating camera · release the look control to aim":"Waiting for a fresh view"):(c?.reason||(lastStatus.mission?"Move the reticle toward a visible object or handle.":"Move the reticle closer to a visible handle."));
    const ready=enabled&&current&&c?.eligible===true;
    $("touch-f").disabled=!ready;$("touch-f").textContent=`F · ${label}`;
    $("action-label").textContent=feedback?feedbackLabels[feedback.state]:label;
    const card=$("interaction");
    for(const state of ["planning","approach","align","failed","cancelled","interacting","done"])card.classList.toggle(`feedback-${state}`,feedback?.state===state);
    if(feedback)Object.assign(card.dataset,{feedbackId:feedback.id,feedbackAction:feedback.action,feedbackState:feedback.state,feedbackAttemptId:String(feedback.attempt_id)});
    else for(const key of ["feedbackId","feedbackAction","feedbackState","feedbackAttemptId"])delete card.dataset[key];
    $("interaction").classList.toggle("ready",ready);
    $("reticle").classList.toggle("ready",ready);$("reticle").classList.toggle("pending",!current);
    const target=$("target-marker");target.hidden=!current||!c;
    if(c){target.style.left=`${100*c.screen_x}%`;target.style.top=`${100*c.screen_y}%`;target.classList.toggle("unavailable",!c.eligible);}
    $("frame-age").textContent=displayed?`${age()>.35?"Delayed · ":""}${fmt(age()*1000,0)} ms`:"No frame";
  }
  function displayControls(){
    $("control-dot").classList.toggle("active",enabled);$("control-label").textContent=enabled?"Controls on":"Controls off";
    $("enable").disabled=enabled||enabling;$("enable").textContent=enabling?"Connecting…":enabled?"Controls enabled":"Enable controls";
    renderCandidate();renderLook();
  }
  function packet(active,axes=[0,0,0],pulse=null){
    const value={token,client_id:client,seq:seq++,ticket,forward:axes[0],lateral:axes[1],yaw:axes[2],interact:Boolean(pulse),active,
      selection_id:pulse?.id||(frameCurrent()?candidate()?.id:null)||null,interaction_candidate_id:pulse?.id||null,
      look_yaw:0,look_heading:look.heading_world,view_mode:look.mode,look_pitch:look.pitch,look_revision:look.revision};
    if(pulse)Object.assign(value,{interaction_frame_seq:pulse.frame_seq,interaction_action:pulse.action,interaction_view_session:pulse.view_session});
    return value;
  }
  function clearFrameBinding(){++frameGeneration;displayed=null;fPulse=null;$("target-marker").hidden=true;for(const key of ["frameSeq","viewRevision","viewSession","selectionId","selectionAction","viewMode","nativeHeading","desiredHeading"])delete $("viewport").dataset[key];$("viewport").dataset.displayed="false";}
  function stop(reason="Walking stopped",send=true){
    ++enableGeneration;enabled=enabling=false;keys.clear();fPulse=null;fHeld=padF=padStop=touchF=mustRelease=false;
    move=[0,0];turn=0;padAxes=[0,0,0];lookStick=[0,0];drag=null;resets.forEach(f=>f());clearFrameBinding();
    mouseCaptured=false;skipCapturedMouse=true;mouseAngleRemaining=0;if(document.pointerLockElement)document.exitPointerLock();
    if(send&&client&&token)json("/api/input",packet(false),true).catch(()=>{});
    activeViewSession=null;displayControls();$("message").textContent=reason;
  }
  async function enable(){
    if(enabled||enabling)return;const generation=++enableGeneration;enabling=true;clearFrameBinding();displayControls();
    try{
      await refresh(true);if(generation!==enableGeneration)return;
      const state=await json("/api/session",{token});if(generation!==enableGeneration)return;
      if(typeof state.view_session!=="string"||!state.look||!Number.isSafeInteger(state.look.revision))throw new Error("Camera session is unavailable");
      client=state.client_id;ticket=state.ticket;seq=state.next_seq;activeViewSession=state.view_session;
      const heading=Number.isFinite(state.look.heading_world)?state.look.heading_world:nativeHeading;
      const mode=state.look.mode||"third_person";
      if(!Number.isFinite(heading)||Math.abs(heading)>Math.PI+1e-6||!Number.isFinite(state.look.pitch)||state.look.pitch< -1.15||state.look.pitch>.60||!["head","third_person"].includes(mode))throw new Error("Camera heading or mode is unavailable");
      // Revision zero must preserve the server's exact floating-point value.
      // Re-wrapping an already bounded heading can change its last bit.
      look={yaw:0,pitch:state.look.pitch,revision:state.look.revision,heading_world:heading,mode};
      // A preserved camera heading need not match the robot after navigation.
      // The fresh lease may also belong to a restarted simulation clock.
      nativeHeading=null;nativeHeadingTime=-Infinity;
      observeHeading(lastStatus.robot_state?.heading??lastStatus.native_heading_world_rad,lastStatus.sim_time);
      keys.clear();fPulse=null;fHeld=touchF=mustRelease=false;padBlocked=true;clearFrameBinding();
      // A release packet arms F. Enable/focus alone never creates an edge.
      await json("/api/input",packet(true));if(generation!==enableGeneration)return;
      if(document.hidden||!document.hasFocus()){stop("Enable controls in the focused page.");return;}
      enabled=true;$("message").textContent="WASD looks · Arrow keys walk · V changes view · F interacts.";
    }catch(e){if(generation===enableGeneration)stop(e.message);}
    finally{if(generation===enableGeneration)enabling=false;displayControls();}
  }
  function setLook(heading,pitch,mode=look.mode){
    if(!enabled||!Number.isFinite(nativeHeading)||!Number.isFinite(heading)||!Number.isFinite(pitch)||!["head","third_person"].includes(mode))return;
    // Camera demand stays continuous through autonomous body turns. The
    // runtime separately bounds the virtual body target and turn speed.
    // Pitch/mode-only edits preserve the exact heading and cannot resume yaw.
    if(heading!==look.heading_world)heading=wrap(heading);
    pitch=clamp(pitch,-1.15,.60);
    if(Math.abs(wrap(heading-look.heading_world))+Math.abs(pitch-look.pitch)<1e-8&&mode===look.mode)return;
    if(look.revision>=Number.MAX_SAFE_INTEGER){stop("Camera revision expired. Enable controls again.");return;}
    look={yaw:0,heading_world:heading,pitch,mode,revision:look.revision+1};fPulse=null;renderCandidate();renderLook();
  }
  function requestInteract(){
    const c=candidate();
    if(enabled&&frameCurrent()&&c?.eligible===true&&typeof c.id==="string"&&INTERACTION_ACTIONS.includes(c.action)&&performance.now()-lastState<=300){
      // Capture the decoded frame's identity once. Never substitute a status target.
      fPulse=Object.freeze({id:c.id,action:c.action,frame_seq:displayed.frame_seq,view_session:displayed.view_session,view_revision:displayed.view_revision,at:performance.now(),frame_age:age()});
    }
  }
  function axes(){
    let f=Number(keys.has("ArrowUp"))-Number(keys.has("ArrowDown"))+move[0]+padAxes[0];
    let l=Number(keys.has("ArrowLeft"))-Number(keys.has("ArrowRight"))+move[1]+padAxes[1];
    f=clamp(f);l=clamp(l);const n=Math.hypot(f,l);if(n>1){f/=n;l/=n;}return[f,l,0];
  }
  async function sendInput(){
    if(!enabled||inputFlight)return;
    if(document.hidden||!document.hasFocus()){stop("Focus lost. Walking stopped.");return;}
    if(performance.now()-lastState>300){stop("Live status expired. Walking stopped.");return;}
    if(fPulse&&(fPulse.frame_age+(performance.now()-fPulse.at)/1000>.30||fPulse.view_session!==activeViewSession||fPulse.view_revision!==look.revision||candidate()?.id!==fPulse.id||candidate()?.action!==fPulse.action))fPulse=null;
    inputFlight=true;const pulse=mustRelease?null:fPulse,generation=enableGeneration;
    try{await json("/api/input",packet(true,axes(),pulse));if(generation===enableGeneration){mustRelease=Boolean(pulse);if(pulse&&fPulse===pulse)fPulse=null;}}
    catch(e){if(generation===enableGeneration)stop(`Input stopped: ${e.message}`);}finally{inputFlight=false;}
  }
  async function refresh(bootstrap=false){
    if(stateFlight&&!bootstrap)return;stateFlight=true;
    try{
      const data=await json(bootstrap?"/api/bootstrap":"/api/state");
      if(data.token)token=data.token;ticket=data.ticket;lastState=performance.now();lastStatus=data.status||{};
      const s=lastStatus,t=data.transport||{};
      observeHeading(s.robot_state?.heading??s.native_heading_world_rad,s.sim_time);
      $("connection").textContent="Connected";$("connection").classList.remove("offline");
      $("mode").textContent=String(s.mode||"waiting").replaceAll("_"," ");
      $("sim-time").textContent=`${fmt(s.sim_time,2)} s`;$("rtf").textContent=`${fmt(s.real_time_factor,2)}×`;
      $("input-age").textContent=t.input_age_s==null?"—":`${fmt(t.input_age_s*1000,0)} ms`;$("controller-ms").textContent=`${fmt(s.controller_ms,1)} ms`;
      if(enabled&&s.message)$("message").textContent=s.message;
      $("opening").hidden=!Number.isFinite(s.drawer_fraction);
      if(Number.isFinite(s.drawer_fraction)){$("drawer-progress").value=clamp(s.drawer_fraction,0,1);$("drawer-percent").textContent=`${fmt(s.drawer_fraction*100,0)}%`;}
      renderMission();
      displayControls();
    }catch(e){$("connection").textContent="Disconnected";$("connection").classList.add("offline");if(enabled)stop("Connection lost. Walking stopped.");}
    finally{stateFlight=false;}
  }
  function validateFrame(v){
    if(!Number.isSafeInteger(v.frame_seq)||v.frame_seq<1||typeof v.view_session!=="string"||!Number.isSafeInteger(v.view_revision)||v.view_revision<0||!Number.isFinite(v.frame_age_s)||v.frame_age_s<0||!Number.isFinite(v.frame_sim_time)||!Number.isInteger(v.width)||!Number.isInteger(v.height)||v.width<160||v.width>2560||v.height<90||v.height>1440||typeof v.jpeg_base64!=="string"||v.jpeg_base64.length>12000000)throw new Error("Invalid camera frame");
    const c=v.selection;if(c!==null&&(!c||typeof c.id!=="string"||!INTERACTION_ACTIONS.includes(c.action)||typeof c.eligible!=="boolean"||![c.screen_x,c.screen_y].every(x=>Number.isFinite(x)&&x>=0&&x<=1)))throw new Error("Invalid frame selection");
    if(!v.camera||!["head","third_person"].includes(v.camera.mode)||![v.camera.native_heading_world_rad,v.camera.desired_heading_world_rad].every(x=>Number.isFinite(x)&&Math.abs(x)<=Math.PI+1e-6))throw new Error("Invalid frame camera heading or mode");
  }
  async function refreshView(){
    if(viewFlight)return;viewFlight=true;const generation=frameGeneration,start=performance.now();let nextURL=null;
    try{
      const v=await json("/api/view");
      if(v.frame_seq===null)return;validateFrame(v);
      if(generation!==frameGeneration)return;
      if(enabled&&v.view_session!==activeViewSession){stop("Camera session changed. Enable controls again.");return;}
      if(activeViewSession&&v.view_session!==activeViewSession)return;
      if(displayed&&v.view_session===displayed.view_session&&v.frame_seq<=displayed.frame_seq)return;
      const raw=atob(v.jpeg_base64),bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);
      nextURL=URL.createObjectURL(new Blob([bytes],{type:"image/jpeg"}));const image=new Image();image.src=nextURL;await image.decode();
      if(image.naturalWidth!==v.width||image.naturalHeight!==v.height)throw new Error("Frame dimensions differ from decoded JPEG");
      await new Promise(resolve=>requestAnimationFrame(()=>{
        if(generation===frameGeneration&&(!activeViewSession||v.view_session===activeViewSession)){
          image.id="stream";image.alt=`Actual G1 ${v.camera.mode==="head"?"head":"third-person"} view from native physics`;image.draggable=false;
          image.dataset.frameSeq=String(v.frame_seq);image.dataset.viewRevision=String(v.view_revision);image.dataset.viewSession=v.view_session;
          $("stream").replaceWith(image);Object.assign($("viewport").dataset,{frameSeq:String(v.frame_seq),viewRevision:String(v.view_revision),viewSession:v.view_session,selectionId:v.selection?.id||"",selectionAction:v.selection?.action||"",viewMode:v.camera.mode,nativeHeading:String(v.camera.native_heading_world_rad),desiredHeading:String(v.camera.desired_heading_world_rad),displayed:"true"});const old=blobURL;blobURL=nextURL;nextURL=null;
          displayed=Object.freeze({frame_seq:v.frame_seq,view_session:v.view_session,view_revision:v.view_revision,selection:v.selection?Object.freeze({...v.selection}):null,camera:Object.freeze({...v.camera}),age_s:v.frame_age_s,request_started:start,frame_sim_time:v.frame_sim_time});
          observeHeading(v.camera.native_heading_world_rad,v.frame_sim_time);
          $("viewport").style.aspectRatio=`${v.width} / ${v.height}`;$("viewport").style.setProperty("--frame-aspect",v.width/v.height);$("waiting").hidden=true;renderCandidate();if(old)URL.revokeObjectURL(old);
        }resolve();
      }));
    }catch(e){$("frame-age").textContent="Camera unavailable";if(displayed&&age()>.35)renderCandidate();}
    finally{if(nextURL)URL.revokeObjectURL(nextURL);viewFlight=false;}
  }
  async function viewLoop(){
    // Sample the 30 Hz native producer on a 60 Hz deadline to avoid losing
    // frames to polling phase. Decode only new IDs, with one request at a time.
    const started=performance.now();await refreshView();
    setTimeout(viewLoop,Math.max(0,VIEW_PERIOD_MS-(performance.now()-started)));
  }
  window.addEventListener("keydown",e=>{
    if(!controlKeys.has(e.code))return;e.preventDefault();
    if(e.code==="Space"||e.code==="Escape"){stop();return;}
    if(e.code==="Enter"||e.code==="NumpadEnter"){if(e.isTrusted&&!e.repeat){$("viewport").focus({preventScroll:true});enable();}return;}
    if(!enabled)return;
    if(e.code==="KeyV"){if(!e.repeat)setLook(look.heading_world,look.pitch,look.mode==="head"?"third_person":"head");return;}
    if(e.code==="KeyF"){if(e.repeat)return;if(!fHeld)requestInteract();fHeld=true;}else{keys.add(e.code);if(!e.repeat)sendInput();}
  });
  window.addEventListener("keyup",e=>{if(!controlKeys.has(e.code))return;e.preventDefault();keys.delete(e.code);if(e.code==="KeyF")fHeld=false;else sendInput();});
  window.addEventListener("blur",()=>stop("Focus lost. Walking stopped."));
  document.addEventListener("visibilitychange",()=>{if(document.hidden)stop("Page hidden. Walking stopped.");});
  window.addEventListener("pagehide",()=>stop("Page closed. Walking stopped."));
  document.addEventListener("pointerlockchange",()=>{
    if(document.pointerLockElement===$("viewport")&&!enabled&&!enabling){
      mouseCaptured=false;document.exitPointerLock();renderLook();return;
    }
    const had=mouseCaptured;mouseCaptured=document.pointerLockElement===$("viewport");drag=null;fPulse=null;
    skipCapturedMouse=true;mouseAngleRemaining=0;
    if(mouseCaptured||had)captureUnavailable=false;
    if(had&&!mouseCaptured)stop("Mouse released. Press Enter for keyboard controls.");else renderLook();
  });
  function captureFailed(){
    if(document.pointerLockElement===$("viewport"))return;
    captureUnavailable=true;mouseCaptured=false;renderLook();
    $("message").textContent="Mouse lock failed. Click “Use mouse look” to retry.";
  }
  document.addEventListener("pointerlockerror",captureFailed);
  $("enable").addEventListener("click",enable);$("stop").addEventListener("click",()=>stop());
  $("reset-look").addEventListener("click",()=>setLook(nativeHeading,-.15));
  $("view-toggle").addEventListener("click",()=>setLook(look.heading_world,look.pitch,look.mode==="head"?"third_person":"head"));
  const viewport=$("viewport");
  function requestMouseCapture(event){
    if(!event.isTrusted)return;
    viewport.focus({preventScroll:true});
    if(!enabled&&!enabling)enable();
    if(document.pointerLockElement===viewport){renderLook();return;}
    captureUnavailable=false;
    if(!viewport.requestPointerLock){captureFailed();return;}
    try{const request=viewport.requestPointerLock();if(request?.catch)request.catch(captureFailed);}catch{captureFailed();}
  }
  $("capture-mouse").addEventListener("click",requestMouseCapture);
  viewport.addEventListener("pointerdown",e=>{
    if(e.button!==0)return;e.preventDefault();viewport.focus({preventScroll:true});
    if(e.pointerType==="mouse"){if(e.isTrusted)enable();return;}
    if(!enabled)return;drag={id:e.pointerId,x:e.clientX,y:e.clientY};viewport.setPointerCapture(e.pointerId);
  });
  document.addEventListener("mousemove",e=>{if(!mouseCaptured||!enabled)return;if(skipCapturedMouse){skipCapturedMouse=false;return;}applyMouseDelta(e.movementX,e.movementY);});
  viewport.addEventListener("pointermove",e=>{if(!drag||e.pointerId!==drag.id||!enabled)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;applyMouseDelta(dx,dy);});
  for(const event of["pointerup","pointercancel","lostpointercapture"])viewport.addEventListener(event,e=>{if(drag?.id===e.pointerId){drag=null;fPulse=null;}});
  const fButton=$("touch-f");fButton.addEventListener("pointerdown",e=>{if(!enabled||fButton.disabled)return;e.preventDefault();fButton.setPointerCapture(e.pointerId);if(!touchF)requestInteract();touchF=true;});
  for(const event of["pointerup","pointercancel","lostpointercapture"])fButton.addEventListener(event,()=>{touchF=false;});
  function stick(id,isLook){
    const el=$(id),knob=el.querySelector("i");let pointer=null;
    const reset=()=>{pointer=null;knob.style.transform="translate(0px,0px)";if(isLook)lookStick=[0,0];else move=[0,0];};resets.push(reset);
    const update=e=>{if(pointer!==e.pointerId||!enabled)return;const r=el.getBoundingClientRect(),radius=Math.min(r.width,r.height)/2-22;let x=(e.clientX-r.left-r.width/2)/radius,y=(e.clientY-r.top-r.height/2)/radius;const n=Math.hypot(x,y);if(n>1){x/=n;y/=n;}knob.style.transform=`translate(${x*radius}px,${y*radius}px)`;if(isLook)lookStick=[-x,-y];else move=[-y,-x];};
    el.addEventListener("pointerdown",e=>{if(!enabled)return;e.preventDefault();pointer=e.pointerId;el.setPointerCapture(pointer);update(e);});el.addEventListener("pointermove",update);
    for(const event of["pointerup","pointercancel","lostpointercapture"])el.addEventListener(event,e=>{if(pointer===e.pointerId)reset();});
  }
  stick("move-stick",false);stick("look-stick",true);
  function turnButton(id,sign){const el=$(id);let pointer=null;el.addEventListener("pointerdown",e=>{if(!enabled)return;e.preventDefault();pointer=e.pointerId;el.setPointerCapture(pointer);turn=sign;});for(const type of["pointerup","pointercancel","lostpointercapture"])el.addEventListener(type,e=>{if(pointer===e.pointerId){pointer=null;turn=0;}});resets.push(()=>{pointer=null;turn=0;});}
  turnButton("turn-left",1);turnButton("turn-right",-1);
  function gamepads(now){
    const dt=clamp((now-lastPadTime)/1000,0,.05);lastPadTime=now;
    mouseAngleRemaining=MOUSE_MAX_RADIANS_PER_SECOND*Math.min(dt,1/30);
    const pad=Array.from(navigator.getGamepads?navigator.getGamepads():[]).find(p=>p?.connected&&p.mapping==="standard");
    let lookX=lookStick[0],lookY=lookStick[1];
    if(pad){
      padName=pad.id;$("gamepad").textContent="Gamepad connected · right stick looks and turns G1 · bumpers turn the view · A interacts · B stops.";
      if(enabled){padAxes=[-dead(pad.axes[1]),-dead(pad.axes[0]),Number(Boolean(pad.buttons[4]?.pressed))-Number(Boolean(pad.buttons[5]?.pressed))];lookX-=dead(pad.axes[2]);lookY-=dead(pad.axes[3]);const down=Boolean(pad.buttons[0]?.pressed),stopping=Boolean(pad.buttons[1]?.pressed);if(!down)padBlocked=false;if(down&&!padF&&!padBlocked)requestInteract();padF=down;if(stopping&&!padStop)stop("Gamepad: walking stopped.");padStop=stopping;}
      else{padAxes=[0,0,0];padF=Boolean(pad.buttons[0]?.pressed);padStop=Boolean(pad.buttons[1]?.pressed);}
    }else{padAxes=[0,0,0];padF=padStop=false;if(padName){padName="";$("gamepad").textContent="Gamepad disconnected.";if(enabled)stop("Gamepad disconnected. Walking stopped.");}}
    const keyboardYaw=Number(keys.has("KeyA")||keys.has("KeyQ"))-Number(keys.has("KeyD")||keys.has("KeyE"));
    const keyboardPitch=Number(keys.has("KeyW"))-Number(keys.has("KeyS"));
    const yawRate=clamp(lookX+turn+padAxes[2])*1.6+keyboardYaw*.8;
    const pitchRate=clamp(lookY)*1.2+keyboardPitch*.6;
    if(enabled&&(yawRate||pitchRate))setLook(look.heading_world+yawRate*dt,look.pitch+pitchRate*dt);
    requestAnimationFrame(gamepads);
  }
  refresh(true);viewLoop();setInterval(()=>refresh(),100);setInterval(sendInput,INPUT_PERIOD_MS);setInterval(renderCandidate,100);requestAnimationFrame(gamepads);
})();
