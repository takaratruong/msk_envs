import xml.etree.ElementTree as ET


def check_muscle_rigid(muscle_element):
    """ set any muscle whose tendon slack length is less than the optimal fiber length as rigid"""
    optimal_fiber_length = float(muscle_element.find("optimal_fiber_length").text)
    tendon_slack_length = float(muscle_element.find("tendon_slack_length").text)
    return tendon_slack_length < optimal_fiber_length


def handle_force_set(force_set):
    # ForceSet -> objects -> forces
    objects = force_set.find("objects")
    new_objects = []

    # Remove all ExponentialContactForce elements
    for force in objects:
        if force.tag == "ExponentialContactForce":
            continue
        elif force.tag == "Millard2012EquilibriumMuscle":
            ignore_tendon_compliance = force.find("ignore_tendon_compliance")
            make_rigid = check_muscle_rigid(force)
            ignore_tendon_compliance.text = "true" if make_rigid else "false"
            print(f"Muscle {force.attrib.get('name', '')} has optimal fiber length {force.find('optimal_fiber_length').text} and tendon slack length {force.find('tendon_slack_length').text}, making it {'rigid' if make_rigid else 'elastic'}")
            new_objects.append(force)
        else:
            new_objects.append(force)
    objects.clear()
    objects.extend(new_objects)
    return


def main():
    base_name = "athlete9_lower"
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(f"{base_name}.osim", parser=parser)
    root = tree.getroot()

    model = root.find("Model")
    force_set = model.find("ForceSet")
    handle_force_set(force_set)

    # Autoformat the XML for better readability
    ET.indent(tree, space="\t")
    tree.write(f"{base_name}_fixed.osim", encoding="utf-8")
    return


if __name__ == "__main__":
    main()
