def generate_code_block(names):
    for name in names:
        print(f'<body name="{name}_ref" pos="0.3 0 1.3">')
        print(
            f'            <geom name="{name}_geoms" type="sphere" size="0.05" rgba="1 0 0 0.1"'
        )
        print(
            '                density="1000000.0" contype="0" conaffinity="0" pos="0.0 0 0.0" />'
        )
        print(
            f'            <joint name="{name}_x" type="slide" axis="1 0 0" range="-1 1" damping="100"/>'
        )
        print("</body>\n")


# Example usage of the function
names = ["pelvis",    "head",      "ltoe",  "rtoe",  "lheel",  "rheel",
    "lknee",     "rknee",     "lhand", "rhand", "lelbow", "relbow",
    "lshoulder", "rshoulder", "lhip",  "rhip"]
generate_code_block(names)