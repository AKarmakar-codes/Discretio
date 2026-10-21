#!/usr/bin/env python3
"""
inspect_datasets.py
Tool for inspecting IDD-PeD and JAAD annotation structures, demographics,
behavioral labels, and bounding box formats for DISCRETIO.
"""

import os
import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd

SAMPLE_DIR = "sample_data"
os.makedirs(SAMPLE_DIR, exist_ok=True)

IDD_PED_XML_URL = "https://raw.githubusercontent.com/Ruthvik9/IDD-PeD/main/data/IDDPedestrian/annotations/gopro/gp_set_0001/gp_set_0001_vid_0001.xml"
JAAD_XML_URL = "https://raw.githubusercontent.com/ykotseruba/JAAD/JAAD_2.0/annotations/video_0001.xml"
JAAD_ATTR_URL = "https://raw.githubusercontent.com/ykotseruba/JAAD/JAAD_2.0/annotations_attributes/video_0001_attributes.xml"


def download_if_missing(url, local_path):
    if not os.path.exists(local_path):
        print(f"Downloading {os.path.basename(local_path)}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(local_path, "wb") as f:
            f.write(resp.read())
    else:
        print(f"Cached: {local_path}")


def inspect_idd_ped():
    xml_path = os.path.join(SAMPLE_DIR, "idd_ped_sample.xml")
    download_if_missing(IDD_PED_XML_URL, xml_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    print("\n" + "=" * 60)
    print("IDD-PeD: Sample Annotation Inspection (gp_set_0001_vid_0001)")
    print("=" * 60)

    # Tracks inspection
    tracks = root.findall(".//track")
    records = []
    poi_metadata = {}

    for t in tracks:
        track_id = t.attrib.get("id")
        label = t.attrib.get("label")

        # Check POI (Pedestrian of Interest) high-level metadata
        if label == "POI":
            boxes = t.findall("box")
            if boxes:
                first_box = boxes[0]
                attrs = {a.attrib.get("name"): a.text for a in first_box.findall("attribute")}
                poi_id = attrs.get("id", track_id)
                poi_metadata[poi_id] = attrs

        if label in ["pedestrian", "POI"]:
            for box in t.findall("box"):
                frame = int(box.attrib.get("frame", 0))
                occluded = box.attrib.get("occluded", "0")
                outside = box.attrib.get("outside", "0")
                if outside == "1":
                    continue
                xtl = float(box.attrib.get("xtl", 0))
                ytl = float(box.attrib.get("ytl", 0))
                xbr = float(box.attrib.get("xbr", 0))
                ybr = float(box.attrib.get("ybr", 0))

                b_attrs = {a.attrib.get("name"): a.text for a in box.findall("attribute")}
                rec = {
                    "dataset": "IDD-PeD",
                    "track_id": track_id,
                    "label": label,
                    "frame": frame,
                    "occluded": occluded,
                    "box": (xtl, ytl, xbr, ybr),
                    "width": round(xbr - xtl, 1),
                    "height": round(ybr - ytl, 1),
                    "attention_intent": b_attrs.get("AI"),
                    "crossing_behavior": b_attrs.get("CB"),
                    "ped_action": b_attrs.get("PA"),
                    "traffic_interaction": b_attrs.get("TI"),
                    "location": b_attrs.get("location"),
                    "joint_interaction": b_attrs.get("Joint Interaction"),
                    "age": b_attrs.get("age"),
                    "gender": b_attrs.get("gender"),
                    "category": b_attrs.get("category"),
                }
                records.append(rec)

    df = pd.DataFrame(records)
    print(f"Total pedestrian/POI bounding boxes extracted: {len(df)}")
    print(f"Unique tracks: {df['track_id'].nunique()}")

    if "age" in df and df["age"].dropna().any():
        print("\nAge distribution:")
        print(df["age"].value_counts(dropna=False))

    print("\nSample Bounding Box & Behavior Records:")
    cols_to_show = ["track_id", "frame", "occluded", "attention_intent", "crossing_behavior", "ped_action", "traffic_interaction", "age"]
    present_cols = [c for c in cols_to_show if c in df.columns]
    print(df[present_cols].dropna(subset=["crossing_behavior", "ped_action"], how="all").head(8).to_string())

    print("\nPOI High-Level Attributes (Sample):")
    for pid, pattrs in list(poi_metadata.items())[:3]:
        print(f"  POI ID: {pid}")
        for k in ["age", "gender", "category", "crossing", "crossing_motive", "road_type", "location_type"]:
            if k in pattrs:
                print(f"     {k}: {pattrs[k]}")

    return df


def inspect_jaad():
    xml_path = os.path.join(SAMPLE_DIR, "jaad_sample.xml")
    attr_path = os.path.join(SAMPLE_DIR, "jaad_attr_sample.xml")
    download_if_missing(JAAD_XML_URL, xml_path)
    download_if_missing(JAAD_ATTR_URL, attr_path)

    print("\n" + "=" * 60)
    print("JAAD: Sample Annotation Inspection (video_0001)")
    print("=" * 60)

    # Parse high level pedestrian attributes
    tree_attr = ET.parse(attr_path)
    ped_attrs = {}
    for p in tree_attr.getroot().findall("pedestrian"):
        pid = p.attrib.get("id")
        ped_attrs[pid] = dict(p.attrib)

    tree = ET.parse(xml_path)
    records = []
    for t in tree.getroot().findall(".//track"):
        label = t.attrib.get("label")
        if label in ["pedestrian", "ped"]:
            boxes = t.findall("box")
            for box in boxes:
                frame = int(box.attrib.get("frame", 0))
                outside = box.attrib.get("outside", "0")
                if outside == "1":
                    continue
                occluded = box.attrib.get("occluded", "0")
                xtl = float(box.attrib.get("xtl", 0))
                ytl = float(box.attrib.get("ytl", 0))
                xbr = float(box.attrib.get("xbr", 0))
                ybr = float(box.attrib.get("ybr", 0))

                b_attrs = {a.attrib.get("name"): a.text for a in box.findall("attribute")}
                rec = {
                    "dataset": "JAAD",
                    "track_label": label,
                    "frame": frame,
                    "occluded": occluded,
                    "box": (xtl, ytl, xbr, ybr),
                    "action": b_attrs.get("action"),
                    "cross": b_attrs.get("cross"),
                    "look": b_attrs.get("look"),
                    "nod": b_attrs.get("nod"),
                }
                records.append(rec)

    df = pd.DataFrame(records)
    print(f"Total pedestrian bounding boxes: {len(df)}")
    print("\nPedestrian Demographic Attributes from JAAD:")
    for pid, pdata in ped_attrs.items():
        print(f"  Ped ID: {pid} -> Age: {pdata.get('age')}, Gender: {pdata.get('gender')}, Crossing: {pdata.get('crossing')}, Designated Crosswalk: {pdata.get('designated')}")

    print("\nSample Box-level Tracking Data (JAAD):")
    cols = ["frame", "occluded", "action", "cross", "look"]
    present = [c for c in cols if c in df.columns]
    print(df[present].head(8).to_string())

    return df


if __name__ == "__main__":
    idd_df = inspect_idd_ped()
    jaad_df = inspect_jaad()
    print("\nInspection complete! Sample files saved to 'sample_data/'.")

