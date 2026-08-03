"""Blender material builders: flat-color and Diffuse/Roughness/Normal-mapped
PBR materials. Imported by blender_scene.py - runs under Blender's bundled
Python (see blender_scene.py's module docstring for that constraint)."""
import bpy


def make_material(name: str, color: tuple, roughness: float = 0.9):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def make_retroreflective_material(name: str, color: tuple, strength: float = 1.6,
                                   roughness: float = 0.15):
    """A material for retroreflective sheeting - hi-vis tape, sign faces.

    Emissive, which is not a cheat. Retroreflective tape returns light back along the
    incident ray, so to any observer roughly behind the light source it is far brighter than
    its diffuse albedo can explain - that is the entire point of it, and it is why a
    delineator post reads at night. EEVEE has no retroreflective BSDF, and a plain white
    diffuse band in a scene lit by one soft sun renders as mid-grey and disappears against
    the post at this camera distance. A little emission puts the band back at the brightness
    a person on the street actually perceives.

    `strength` is a look control, not a photometric quantity - there is no cd/lx/m^2 figure
    behind it. Kept modest so the band reads as bright tape rather than a light source.
    """
    mat = make_material(name, color, roughness)
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    # Blender renamed this socket: "Emission Color" from 4.0, plain "Emission" before it.
    emission = bsdf.inputs.get("Emission Color") or bsdf.inputs.get("Emission")
    if emission is not None:
        emission.default_value = (*color, 1.0)
        bsdf.inputs["Emission Strength"].default_value = strength
    return mat


def make_textured_material(name: str, texture_paths: dict | None, fallback_color: tuple,
                            fallback_roughness: float = 0.9):
    """Diffuse/Roughness/Normal-mapped material from local file paths (already
    downloaded by src/render/theme.py in the venv - this function never fetches
    anything). Falls back to a flat color material if texture_paths is falsy
    or any image fails to load, so a missing/corrupt file never crashes the render."""
    if not texture_paths:
        return make_material(name, fallback_color, fallback_roughness)
    try:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        bsdf = nodes.get("Principled BSDF")

        def image_node(path: str, colorspace: str):
            node = nodes.new("ShaderNodeTexImage")
            img = bpy.data.images.load(path)
            img.colorspace_settings.name = colorspace
            node.image = img
            return node

        if texture_paths.get("Diffuse"):
            links.new(image_node(texture_paths["Diffuse"], "sRGB").outputs["Color"], bsdf.inputs["Base Color"])
        if texture_paths.get("Rough"):
            links.new(image_node(texture_paths["Rough"], "Non-Color").outputs["Color"], bsdf.inputs["Roughness"])
        if texture_paths.get("nor_gl"):
            normal_map = nodes.new("ShaderNodeNormalMap")
            links.new(image_node(texture_paths["nor_gl"], "Non-Color").outputs["Color"], normal_map.inputs["Color"])
            links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
        return mat
    except Exception as e:
        print(f"  WARNING: textured material {name!r} failed ({e}) - falling back to flat color")
        return make_material(name, fallback_color, fallback_roughness)
