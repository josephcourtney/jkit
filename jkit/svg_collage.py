#!/usr/bin/env python
# coding:utf-8

import xml.etree.cElementTree as ET
from xml.dom import minidom
import base64

dpi = 72


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")


def init_svg(width=7, height=4):
    svg = ET.Element("svg")
    svg.set("xmlns", "http://www.w3.org/2000/svg")
    svg.set("version", "1.1")
    svg.set("xmlns:xlink", "http://www.w3.org/1999/xlink")
    svg.set("width", str(width * dpi))
    svg.set("height", str(height * dpi))
    return svg


def add_sub_plot(svg, path, translate=(0, 0), scale=None, rotate=None):
    defs = svg.find("defs")
    if defs is None:
        defs = ET.SubElement(svg, "defs")
    groups = defs.findall("g")
    if len(groups) == 0:
        idx = 0
    else:
        idx = 1 + max([int(g.attrib["id"].split("_")[-1]) for g in groups])
    tree = ET.parse(path)
    root = tree.getroot()
    root.tag = "g"
    root.set("id", f"input_svg_{idx}")
    defs.insert(-1, root)

    use = ET.SubElement(svg, "use")
    xfrm = []
    if translate is not None:
        xfrm.append(f"translate({translate[0]} {translate[1]})")
    if rotate is not None:
        xfrm.append(f"rotate({rotate})")
    if scale is not None:
        xfrm.append(f"scale({scale[0]} {scale[1]})")
    use.set("transform", " ".join(xfrm))
    use.set("xlink:href", f"#input_svg_{idx}")
    return svg


def embed_image(svg, img_path, translate=None, scale=None, rotate=None):
    img = ET.SubElement(svg, "image")
    xfrm = []
    if translate is not None:
        xfrm.append(f"translate({translate[0]} {translate[1]})")
    if rotate is not None:
        xfrm.append(f"rotate({rotate})")
    if scale is not None:
        xfrm.append(f"scale({scale[0]} {scale[1]})")
    if rotate is not None or scale is not None or translate is not None:
        img.set("transform", " ".join(xfrm))
    encoded = base64.b64encode(open(img_path, "rb").read()).decode("utf-8")
    img.set("xlink:href", f"data:image/png;base64,{encoded}")
    return svg


def annotation(
    svg,
    txt,
    pos=None,
    translate=None,
    scale=None,
    rotate=None,
    font_family="Helvetica",
    font_size=12,
    font_weight="regular",
    color="#000000",
    style="",
    attrs=None,
):
    text = ET.SubElement(svg, "text")
    if pos is not None:
        text.set("x", str(pos[0]))
        text.set("y", str(pos[1]))
    xfrm = []
    if translate is not None:
        xfrm.append(f"translate({translate[0]} {translate[1]})")
    if rotate is not None:
        xfrm.append(f"rotate({rotate})")
    if scale is not None:
        xfrm.append(f"scale({scale[0]} {scale[1]})")
    if translate is not None or rotate is not None or scale is not None:
        text.set("transform", " ".join(xfrm))
    style = f"{style}"
    style += f"font-family:{font_family};"
    style += f"font-size:{font_size}pt;"
    style += f"font-weight:{font_weight};"
    style += f"fill:{color};"
    text.set("style", style)

    if attrs is not None:
        for k, v in attrs.items():
            text.set(k, v)
    text.text = txt
    return svg


def add_element(
    svg, tag, pos=None, translate=None, scale=None, rotate=None, attrs=None
):
    elm = ET.SubElement(svg, tag)
    if pos is not None:
        elm.set("x", str(pos[0]))
        elm.set("y", str(pos[1]))
    xfrm = []
    if translate is not None:
        xfrm.append(f"translate({translate[0]} {translate[1]})")
    if rotate is not None:
        xfrm.append(f"rotate({rotate})")
    if scale is not None:
        xfrm.append(f"scale({scale[0]} {scale[1]})")
    if rotate is not None or scale is not None or translate is not None:
        elm.set("transform", " ".join(xfrm))
    if attrs is not None:
        for k, v in attrs.items():
            elm.set(k, str(v))
    return svg
