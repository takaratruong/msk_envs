import xml.etree.ElementTree as ElementTree
from msk_envs.utils.osim_objs import *


def to_vector2(text: str) -> Vector2:
    v2 = tuple(map(float, text.split()))
    assert (len(v2) == 2)
    return Vector2(x=v2[0], y=v2[1])


def to_vector3(text: str) -> Vector3:
    v3 = tuple(map(float, text.split()))
    assert (len(v3) == 3)
    return Vector3(x=v3[0], y=v3[1], z=v3[2])


def to_vector6(text: str) -> Vector6:
    v6 = tuple(map(float, text.split()))
    assert (len(v6) == 6)
    return Vector6(v0=v6[0], v1=v6[1], v2=v6[2], v3=v6[3], v4=v6[4], v5=v6[5])


def parse_coordinate(coordinate) -> Coordinate:
    name = coordinate.attrib["name"]
    motion_type = MotionType(coordinate.find("motion_type").text)
    default_value = float(coordinate.find("default_value").text)
    default_speed_value = float(coordinate.find("default_speed_value").text)
    range_text = coordinate.find("range").text
    range_values = to_vector2(range_text)
    clamped_text = coordinate.find("clamped").text
    clamped = clamped_text.lower() == "true"
    locked_text = coordinate.find("locked").text
    locked = locked_text.lower() == "true"

    prescribed_text = coordinate.find("prescribed").text
    prescribed = prescribed_text.lower() == "true"
    assert (not prescribed), "Prescribed coordinates are not supported"
    return Coordinate(
        name=name,
        motion_type=motion_type,
        default_value=default_value,
        default_speed_value=default_speed_value,
        range=range_values,
        clamped=clamped,
        locked=locked
    )


def parse_coordinate_set(coordinate_set) -> CoordinateSet:
    coordinates = OrderedDict()
    coordinate_set_objects = coordinate_set.find("objects")
    coordinate_set_coordinates = coordinate_set_objects.findall("Coordinate")
    for coordinate in coordinate_set_coordinates:
        coord_obj = parse_coordinate(coordinate)
        coordinates[coord_obj.name] = coord_obj
    return CoordinateSet(coordinates=coordinates)


def parse_function(function) -> Function:
    for child in function:
        if child.tag == "LinearFunction":
            coefficients = to_vector2(child.find("coefficients").text)
            return LinearFunction(coefficients=coefficients)
        elif child.tag == "Constant":
            value = float(child.find("value").text)
            return ConstantFunction(value=value)
        elif child.tag == "MultiplierFunction":
            inner_function = parse_function(child.find("function"))
            scale = float(child.find("scale").text)
            inner_function.scale(scale)
            return inner_function
        else:
            raise NotImplementedError(
                f"Function type {child.tag} not supported")
    raise ValueError("No function found")


def parse_transform_axis(transform_axis) -> TransformAxis:
    name = transform_axis.attrib["name"]
    coordinates = transform_axis.find("coordinates").text
    axis = to_vector3(transform_axis.find("axis").text)
    function = parse_function(transform_axis.find("function"))
    return TransformAxis(
        name=name,
        coordinates=coordinates,
        axis=axis,
        function=function
    )


def parse_spatial_transform(spatial_transform) -> SpatialTransform:
    transform_axes = []
    spatial_transform_objects = spatial_transform.findall("TransformAxis")
    for ta in spatial_transform_objects:
        ta_obj = parse_transform_axis(ta)
        transform_axes.append(ta_obj)

    return SpatialTransform(transform_axes=transform_axes)


def parse_custom_joint(joint) -> Joint:
    name = joint.attrib["name"]
    parent_body = joint.find("parent_body").text
    location_in_parent = to_vector3(joint.find("location_in_parent").text)
    orientation_in_parent = to_vector3(joint.find("orientation_in_parent").text)
    location = to_vector3(joint.find("location").text)
    orientation = to_vector3(joint.find("orientation").text)
    coordinate_set_element = joint.find("CoordinateSet")
    coordinate_set = parse_coordinate_set(coordinate_set_element)

    # SpatialTransform describes how body moves wrst parent
    spatial_transform_element = joint.find("SpatialTransform")
    spatial_transform = parse_spatial_transform(spatial_transform_element)

    # we don't support this
    reverse = joint.find("reverse")
    assert (reverse is None or reverse.text.lower() == "false")

    return CustomJoint(
        name=name,
        parent_body=parent_body,
        location_in_parent=location_in_parent,
        orientation_in_parent=orientation_in_parent,
        location=location,
        orientation=orientation,
        coordinate_set=coordinate_set,
        spatial_transform=spatial_transform
    )


def parse_display_geometry(display_geometry):
    geometry_file = display_geometry.find("geometry_file").text
    # replace .vtp with .obj
    if geometry_file.endswith(".vtp"):
        geometry_file = geometry_file[:-4] + ".obj"
    color = to_vector3(display_geometry.find("color").text)
    texture_file_element = display_geometry.find("texture_file")
    texture_file = texture_file_element.text if (
            texture_file_element is not None) else None
    transform = to_vector6(display_geometry.find("transform").text)
    scale_factors = to_vector3(display_geometry.find("scale_factors").text)
    return DisplayGeometry(
        geometry_file=geometry_file,
        color=color,
        texture_file=texture_file,
        transform=transform,
        scale_factors=scale_factors
    )


def parse_body_mass(body_element) -> BodyMass:
    body_mass = body_element.find("mass")
    body_mass_center = body_element.find("mass_center")
    body_inertia_xx = body_element.find("inertia_xx")
    body_inertia_yy = body_element.find("inertia_yy")
    body_inertia_zz = body_element.find("inertia_zz")
    body_inertia_xy = body_element.find("inertia_xy")
    body_inertia_xz = body_element.find("inertia_xz")
    body_inertia_yz = body_element.find("inertia_yz")

    return BodyMass(
        mass=float(body_mass.text),
        mass_center=to_vector3(body_mass_center.text),
        inertia=Inertia(
            xx=float(body_inertia_xx.text),
            yy=float(body_inertia_yy.text),
            zz=float(body_inertia_zz.text),
            xy=float(body_inertia_xy.text),
            xz=float(body_inertia_xz.text),
            yz=float(body_inertia_yz.text),
        ),
    )


def parse_visible_object(visible_object) -> VisibleObject:
    geometry_set = visible_object.find("GeometrySet")
    geometry_set_objects = geometry_set.find("objects")
    objects_display = geometry_set_objects.findall("DisplayGeometry")
    display_geometries = []
    for obj in objects_display:
        display_geometry = parse_display_geometry(obj)
        display_geometries.append(display_geometry)

    # Three scale factors: scaleX, scaleY, scaleZ
    scale_factors = visible_object.find("scale_factors")
    scale_factors_v3 = to_vector3(scale_factors.text)

    # Transform relative to owner specified as 3 rotations and 3 translations
    transform = visible_object.find("transform")
    transform_v6 = to_vector6(transform.text)
    return VisibleObject(
        geometry_set=display_geometries,
        scale_factors=scale_factors_v3,
        transform=transform_v6
    )


def parse_body_set(body_set) -> BodySet:
    bodies = OrderedDict()
    body_set_objects = body_set.find("objects")
    body_set_bodies = body_set_objects.findall("Body")
    for body in body_set_bodies:
        body_name = body.attrib["name"]
        # All mass attributes
        body_mass = parse_body_mass(body)
        # Visible objects
        visible_object_element = body.find("VisibleObject")
        body_visible_object = parse_visible_object(visible_object_element)
        # Joint (connecting to parent body)
        joint_element = body.find("Joint")
        if list(joint_element):
            # only CustomJoint is supported for now
            custom_joint_element = joint_element.find("CustomJoint")
            joint = parse_custom_joint(custom_joint_element)
        else:
            joint = None

        # Create Body object
        body_obj = Body(
            name=body_name,
            body_mass=body_mass,
            visible_object=body_visible_object,
            joint=joint
        )
        bodies[body_name] = body_obj
    return BodySet(bodies=bodies)


def parse_marker(marker) -> Marker:
    name = marker.attrib["name"]
    body = marker.find("body").text
    location = to_vector3(marker.find("location").text)
    fixed_text = marker.find("fixed").text
    fixed = fixed_text.lower() == "true"
    return Marker(
        name=name,
        body=body,
        location=location,
        fixed=fixed
    )


def parse_marker_set(marker_set) -> MarkerSet:
    markers = OrderedDict()
    marker_set_objects = marker_set.find("objects")
    marker_set_markers = marker_set_objects.findall("Marker")
    for marker in marker_set_markers:
        marker_obj = parse_marker(marker)
        markers[marker_obj.name] = marker_obj
    return MarkerSet(markers=markers)


def parse_path_point(path_point) -> PathPoint:
    name = path_point.attrib["name"]
    body = path_point.find("body").text
    location = to_vector3(path_point.find("location").text)
    return PathPoint(
        name=name,
        body=body,
        location=location)


def parse_conditional_path_point(
        conditional_path_point) -> ConditionalPathPoint:
    name = conditional_path_point.attrib["name"]
    body = conditional_path_point.find("body").text
    location = to_vector3(conditional_path_point.find("location").text)
    range_text = conditional_path_point.find("range").text
    range_values = to_vector2(range_text)
    coordinate = conditional_path_point.find("coordinate").text
    return ConditionalPathPoint(
        name=name,
        body=body,
        location=location,
        range=range_values,
        coordinate=coordinate
    )


def parse_path_point_set(path_point_set) -> PathPointSet:
    path_points = OrderedDict()
    path_point_set_objects = path_point_set.find("objects")
    for child in path_point_set_objects:
        if child.tag == "PathPoint":
            pp_obj = parse_path_point(child)
            path_points[pp_obj.name] = pp_obj
        elif child.tag == "ConditionalPathPoint":
            cpp_obj = parse_conditional_path_point(child)
            path_points[cpp_obj.name] = cpp_obj
    return PathPointSet(path_points=path_points)


def parse_geometry_path(geometry_path) -> GeometryPath:
    path_point_set_element = geometry_path.find("PathPointSet")
    path_point_set = parse_path_point_set(path_point_set_element)
    return GeometryPath(path_point_set=path_point_set)


def parse_muscle(muscle) -> Muscle:
    name = muscle.attrib["name"]
    # GeometryPath
    geometry_path_element = muscle.find("GeometryPath")
    geometry_path = parse_geometry_path(geometry_path_element)

    # Muscle properties
    max_isometric_force = float(muscle.find("max_isometric_force").text)
    optimal_fiber_length = float(muscle.find("optimal_fiber_length").text)
    tendon_slack_length = float(muscle.find("tendon_slack_length").text)
    pennation_angle_at_optimal = float(
        muscle.find("pennation_angle_at_optimal").text)
    return Muscle(
        name=name,
        geometry_path=geometry_path,
        max_isometric_force=max_isometric_force,
        optimal_fiber_length=optimal_fiber_length,
        tendon_slack_length=tendon_slack_length,
        pennation_angle_at_optimal=pennation_angle_at_optimal
    )

def parse_actuator(actuator) -> Actuator:
    name = actuator.attrib["name"]
    optimal_force = float(actuator.find("optimal_force").text)
    coordinate = actuator.find("coordinate").text
    return Actuator(
        name=name,
        optimal_force=optimal_force,
        coordinate=coordinate
    )

def parse_force_set(force_set) -> ForceSet:
    muscles = OrderedDict()
    force_set_objects = force_set.find("objects")
    force_set_muscles = force_set_objects.findall("Thelen2003Muscle")
    for muscle in force_set_muscles:
        muscle_obj = parse_muscle(muscle)
        muscles[muscle_obj.name] = muscle_obj

    actuators = OrderedDict()
    force_set_actuators = force_set_objects.findall("ActivationCoordinateActuator")
    for actuator in force_set_actuators:
        actuator_obj = parse_actuator(actuator)
        actuators[actuator_obj.name] = actuator_obj

    return ForceSet(muscles=muscles, actuators=actuators)


def parse_contact_half_space(contact_half_space) -> ContactHalfSpace:
    name = contact_half_space.attrib["name"]
    body_name = contact_half_space.find("body_name").text
    location = to_vector3(contact_half_space.find("location").text)
    orientation = to_vector3(contact_half_space.find("orientation").text)
    return ContactHalfSpace(
        name=name,
        body_name=body_name,
        location=location,
        orientation=orientation
    )


def parse_contact_sphere(contact_sphere) -> ContactSphere:
    name = contact_sphere.attrib["name"]
    body_name = contact_sphere.find("body_name").text
    location = to_vector3(contact_sphere.find("location").text)
    orientation = to_vector3(contact_sphere.find("orientation").text)
    radius = float(contact_sphere.find("radius").text)
    return ContactSphere(
        name=name,
        body_name=body_name,
        location=location,
        orientation=orientation,
        radius=radius
    )


def parse_contact_geometry_set(contact_geometry_set) -> ContactGeometrySet:
    contact_half_spaces = OrderedDict()
    contact_spheres = OrderedDict()
    contact_geometry_set_objects = contact_geometry_set.find("objects")
    if contact_geometry_set_objects is not None:
        contact_half_space_elements = contact_geometry_set_objects.findall(
            "ContactHalfSpace")
        for chs in contact_half_space_elements:
            chs_obj = parse_contact_half_space(chs)
            contact_half_spaces[chs_obj.name] = chs_obj

        contact_sphere_elements = contact_geometry_set_objects.findall(
            "ContactSphere")
        for cs in contact_sphere_elements:
            cs_obj = parse_contact_sphere(cs)
            contact_spheres[cs_obj.name] = cs_obj

    return ContactGeometrySet(
        contact_half_spaces=contact_half_spaces,
        contact_spheres=contact_spheres
    )


def parse_osim_file(file_path: str) -> Model:
    """
    Model ->
        BodySet ->
            Bodies
        Constraints
        Forces
        Controllers
    """
    tree = ElementTree.parse(file_path)
    root = tree.getroot()
    # Everything is under <Model>
    model_element = root.find("Model")

    # BodySet
    body_set_element = model_element.find("BodySet")
    body_set = parse_body_set(body_set_element)

    # ForceSet
    force_set_element = model_element.find("ForceSet")
    force_set = parse_force_set(force_set_element)

    # MarkerSet
    marker_set_element = model_element.find("MarkerSet")
    marker_set = parse_marker_set(marker_set_element)

    # ContactGeometrySet
    contact_geometry_set_element = model_element.find("ContactGeometrySet")
    contact_geometry_set = parse_contact_geometry_set(
        contact_geometry_set_element)

    # The remaining are unsupported
    constraint_set = model_element.find("ConstraintSet")
    component_set = model_element.find("ComponentSet")
    controller_set = model_element.find("ControllerSet")

    # Create Model object
    model = Model(
        body_set=body_set,
        force_set=force_set,
        marker_set=marker_set,
        contact_geometry_set=contact_geometry_set
    )
    return model


if __name__ == "__main__":
    # Example usage
    osim_file_path = "Scaled_FullBody_HamnerModel_Muscle_withContact.osim"
    parse_osim_file(osim_file_path)
