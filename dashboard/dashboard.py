import argparse
import csv
import gzip
import json
import os
import re
from pathlib import Path
from nicegui import ui, app, events
from timeline_controller import TimelineController

# Parse command-line arguments (only when run directly)
if __name__ in {"__main__", "__mp_main__"}:
    parser = argparse.ArgumentParser(description='MSK Dashboard')
    parser.add_argument('--traj-dir', type=str, default='trajectories',
                        help='Trajectory directory (default: trajectories)')
    args = parser.parse_args()
    TRAJ_DIR = args.traj_dir
else:
    TRAJ_DIR = 'trajectories'


def dispatch_custom_event(event_name: str, data: str):
    """ Helper for custom events to viewer or any listeners """
    ui.run_javascript(f'''
        window.dispatchEvent(new CustomEvent('{event_name}', {{
            detail: {data}
        }}));
    ''')


def format_timeline_label():
    value = timeline_slider.value
    end_value = timeline_slider.props['max']
    slider_value_label.set_text(f"Frame {value} of {end_value}")
    return


def reset_slider():
    timeline_slider.value = 0
    format_timeline_label()


def reset_viewer():
    dispatch_custom_event('resetViewer', '{}')


def reset_dashboard():
    reset_viewer()
    reset_slider()


def locate_traj_dirs():
    traj_dir_options.clear()
    # Look for all subdirectories in trajectories/
    dirs = [d for d in os.listdir(TRAJ_DIR) if os.path.isdir(os.path.join(TRAJ_DIR, d))]
    dirs = sorted(dirs)

    with traj_dir_options:
        # Add option to show all trajectories
        ui.item("All", on_click=lambda: select_traj_dir(""))
        for dir_name in dirs:
            ui.item(dir_name, on_click=lambda d=dir_name: select_traj_dir(d))
    return


def select_traj_dir(dir_name: str):
    global selected_traj_dir
    selected_traj_dir = dir_name
    traj_dir_label.set_text(f"Dir: {dir_name if dir_name else 'All'}")
    return


def locate_trajectories():
    trajectory_options.clear()

    base_path = (
        Path(TRAJ_DIR) / selected_traj_dir
        if selected_traj_dir
        else Path(TRAJ_DIR)
    )

    # Find .json and .json.gz files
    traj_files = list(base_path.rglob("*.json")) + list(base_path.rglob("*.json.gz"))

    def sort_key(path: Path):
        """Sort by directory, then by filename (natural sorting)"""
        dir_name = str(path.parent)

        name = path.name
        if name.endswith(".json.gz"):
            file_name = name[:-8]  # strip ".json.gz"
        else:
            file_name = path.stem  # strips ".json"

        # Extract numeric part for natural sorting
        numbers = re.findall(r'\d+', file_name)
        numeric_part = int(numbers[0]) if numbers else 0

        return (dir_name, numeric_part, file_name)

    # Sort by directory then by numeric value, then by filename
    traj_files = sorted(traj_files, key=sort_key)

    with trajectory_options:
        for path in traj_files:
            name = str(path.relative_to(TRAJ_DIR))
            ui.item(name, on_click=lambda p=path: send_viewer_trajectory(str(p)))
    return


def send_viewer_trajectory(file: str):
    reset_dashboard()

    if file.endswith(".gz"):
        with gzip.open(file, 'rt') as f:
            traj = json.load(f)
    else:
        traj = json.loads(open(file).read())

    # update slider
    n_frames = len(traj)
    current_traj_label.set_text(file)
    timeline_controller.max_ticks = n_frames - 1
    timeline_slider.props['max'] = n_frames - 1
    timeline_slider.update()
    format_timeline_label()

    # update fps
    if len(traj) > 1:
        fps = 1.0 / (traj[1]["time"] - traj[0]["time"])
    else:
        fps = 1.0 / traj[0]["time"]
    timeline_controller.set_tick_rate(fps)
    tick_rate_input.value = int(fps)
    tick_rate_input.update()

    # send to viewer
    dispatch_custom_event('loadTrajectory', traj)


def send_inference_results(file: str):
    iteration = []
    data = []
    with open(file, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            iteration.append(int(row[0]))
            data.append(list(map(float, row[1:])))


def slider_change(e: events.ValueChangeEventArguments):
    """Handle slider value changes"""
    current_value = e.value
    timeline_controller.slider_change(current_value)
    format_timeline_label()
    dispatch_custom_event('sliderChanged', current_value)


def toggle_play():
    timeline_controller.toggle_playback()
    return


def set_tick_rate(value):
    timeline_controller.set_tick_rate(value)
    return


app.add_static_files('/css', 'css')
app.add_static_files('/js', 'js')
app.add_static_files('/trajectories', TRAJ_DIR)

app.add_static_files('/assets', 'assets')
app.add_static_files('/assets/geometry', 'assets/geometry')
app.add_static_files('/assets/textures', 'assets/textures')

# Header import three js
head_html = f"""
<script type="importmap">
    {{
      "imports": {{
        "three": "https://cdn.jsdelivr.net/npm/three@0.179.1/build/three.module.js",
        "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.179.1/examples/jsm/",
        "plotly": "https://cdn.plot.ly/plotly-2.34.0.min.js"
      }}
    }}
</script>
<script type="module" src="js/viewer.js"></script>
<link rel="stylesheet" href="css/main.css">
"""
ui.add_head_html(head_html)

ui.add_head_html('')
ui.colors(primary='#555')

# Initialize selected trajectory directory
selected_traj_dir = ""

with ui.row().classes('w-full z-50 bg-white').style(
        'border: 3px solid #ccc; border-radius: 8px; box-sizing: border-box;'):
    # Title, centered
    with ui.row().classes("w-full justify-center mt-4"):
        ui.label("MSK Dashboard").classes("text-2xl font-bold")
        with ui.row().classes("items-center justify-center gap-4"):
            traj_dir_options = ui.dropdown_button("Traj Dir",
                                                  auto_close=True,
                                                  on_click=locate_traj_dirs)
            traj_dir_label = ui.label("Dir: All")
            trajectory_options = ui.dropdown_button("Trajectories",
                                                    auto_close=True,
                                                    on_click=locate_trajectories)

    # Set up the renderer/viewer
    ui.html("""
        <div style="width: 100%">
            <div id="viewer"></div>
            <div class="controls" style="padding: 1em;">
                <div class="control-row">
                    <label for="drawVisuals">Draw visuals</label>
                    <input type="checkbox" id="drawVisuals" checked>

                    <label for="drawCapsuleColliders">Draw capsule colliders</label>
                    <input type="checkbox" id="drawCapsuleColliders">
                    
                    <label for="drawSphereColliders">Draw sphere colliders</label>
                    <input type="checkbox" id="drawSphereColliders">

                    <label for="drawMuscles">Draw muscles</label>
                    <input type="checkbox" id="drawMuscles" checked>
                </div>
                <button id="fullScreenButton">Full Screen</button>
                <div class="control-row">
                    <button id="resetButton1">Camera 1 follow</button>
                    <button id="resetButton2">Camera 2 follow</button>
                    <button id="resetButton3">Camera 3 follow</button>
                </div>
            </div>
        </div>
    """, sanitize=False).classes('w-full')

    # Control panel
    with ui.column().classes("items-center w-full"):
        timeline_slider = ui.slider(min=0, max=0, value=0, on_change=slider_change)
        timeline_slider.props('id="timeline"')
        with ui.row().classes("items-center"):
            current_traj_label = ui.label("")
            slider_value_label = ui.label("")
            time_value_label = ui.label("").props('id="timeValue"')
            play_button = ui.button(
                "Play",
                on_click=toggle_play
            )
            ui.label("Tick Rate:")
            tick_rate_input = ui.number(
                value=15,
                min=0.0,
                step=1.0,
                on_change=lambda e: set_tick_rate(e.value)
            ).classes("w-20")

timeline_controller = TimelineController(tick_rate=tick_rate_input.value,
                                         max_ticks=timeline_slider.props["max"],
                                         play_button=play_button,
                                         timeline_slider=timeline_slider)

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(show=False, title="MSK Dashboard",
           favicon="assets/textures/favicon.png")
