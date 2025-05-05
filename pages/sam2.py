from collections import defaultdict
import io
import json
import time
import tempfile
import os
from typing import Iterator, List

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas

from clarifai.client import Model
from clarifai.runners.utils import data_types as dt

st.set_page_config(layout="wide")
st.title("🎥 SAM2 Object Tracking Demo")

def object_to_region(obj)-> List[dt.Region]:
  points = obj.get("points", [])
  labels = obj.get("labels", [])
  regions = []
  for ((x, y), lb) in zip(points, labels):
    lb = int(lb)
    reg = dt.Region(concepts=[dt.Concept(name=str(lb), value=float(lb))])
    reg.proto.region_info.point.col = x
    reg.proto.region_info.point.row = y
    
    regions.append(reg)
    
  return regions


def objects_to_frames(objs):
  frames_ = defaultdict(lambda : [])
  for obj in objs:
    points = obj.get("points", [])
    labels = obj.get("labels", [])
    frame_idx = obj.get("frame_idx")
    obj_id = obj.get("obj_id")
    regions = []
    for ((x, y), lb) in zip(points, labels):
      lb = int(lb)
      reg = dt.Region(concepts=[dt.Concept(name=str(lb), value=float(lb))])
      reg.proto.region_info.point.col = x
      reg.proto.region_info.point.row = y
      reg.proto.track_id = str(obj_id)
      regions.append(reg)
    frames_[frame_idx] += regions
  frames = []
  for (idx, fs) in frames_.items():
    frame = dt.Frame(regions=fs)
    frame.proto.frame_info.index = idx
    frames.append(frames)
  print(frames)
  return frames

def display():
  # -------------------- Side bar
  # Sidebar
  st.sidebar.header("Model Setting")
  model_url = st.sidebar.text_input(
      "Model URL", 
      value="https://clarifai.com/meta/segment-anything/models/sam2_1-hiera-base-plus")
  base_url = st.sidebar.text_input("Base URL", value=os.environ.get("CLARIFAI_API_BASE","https://api.clarifai.com"))
  if base_url:
    os.environ["CLARIFAI_API_BASE"] = base_url
  pat = st.sidebar.text_input("PAT", type="password")
  if pat:
    os.environ["CLARIFAI_PAT"] = pat
  if not os.environ.get("CLARIFAI_PAT"):
    st.error("Please insert your PAT")
    st.stop()

  # Init model
  if not "model_url" in st.session_state or st.session_state["model_url"] != model_url:
    st.session_state["model_url"] = model_url
  model = Model(url=st.session_state["model_url"], pat=os.environ.get(
      "CLARIFAI_PAT", "xx"), base_url=base_url)
  st.subheader(f"Model ID: `{str(model.id)}`")
  with st.sidebar.expander("`Runner selector`", expanded=True):
    user_id = st.text_input("user_id", model.user_app_id.user_id)
    deployment_id = st.text_input("deployment_id", os.environ.get("CLARIFAI_DEPLOYMENT_ID") )
    compute_cluster_id = st.text_input(
        "compute_cluster_id", os.environ.get("CLARIFAI_COMPUTE_CLUSTER_ID"))
    nodepool_id = st.text_input(
        "nodepool_id", os.environ.get("CLARIFAI_NODEPOOL_ID"))
    model._set_runner_selector(
        compute_cluster_id=compute_cluster_id,
        nodepool_id=nodepool_id,
        deployment_id=deployment_id,
        user_id=user_id,
    )
    #print(model)

  # ------------------- SESSION STATE INIT ----------------------
  if "objects" not in st.session_state:
      st.session_state.objects = []
  if "frame_idx" not in st.session_state:
      st.session_state.frame_idx = 0
  if "rendered_frame_idx" not in st.session_state:
      st.session_state.rendered_frame_idx = 0
  if "video_frames" not in st.session_state:
      st.session_state.video_frames = []
  if "obj_id" not in st.session_state:
      st.session_state.obj_id = 0
  if "current_mode" not in st.session_state:
      st.session_state.current_mode = "positive"
  if "current_video" not in st.session_state:
      st.session_state.current_video = None
  if "obj_to_color" not in st.session_state:
      st.session_state.obj_to_color = {}
  if "canvas_img" not in st.session_state:
      st.session_state.canvas_img = None
  if "fps" not in st.session_state:
      st.session_state.fps = 0
  

  # ------------------- UPLOAD VIDEO ----------------------------
  uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
  if uploaded_file and uploaded_file.name != st.session_state.current_video:
      st.session_state.current_video = uploaded_file.name
      tfile = tempfile.NamedTemporaryFile(delete=True, suffix=".mp4", prefix="input-sam2-")
      tfile.write(uploaded_file.read())
      print(tfile.name)
      cap = cv2.VideoCapture(tfile.name)
      frames = []
      while True:
          ret, frame = cap.read()
          if not ret:
              break
          frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
      cap.release()
      tfile.close()
      #os.remove(tfile.name)
      st.session_state.video_frames = frames
      st.session_state.frame_idx = 0

  if st.button("Reset everything", type="primary"):
      st.session_state.setdefault("objects", [])
      st.session_state.setdefault("frame_idx", 0)
      st.session_state.setdefault("rendered_frame_idx", 0)
      st.session_state.setdefault("video_frames", [])
      st.session_state.setdefault("obj_id", 0)
      st.session_state.setdefault("current_mode", "positive")
      st.session_state.setdefault("current_video", None)
      st.session_state.setdefault("obj_to_color", {})
      st.session_state.setdefault("canvas_img", None)
      st.session_state.setdefault("fps", 0)
      st.session_state.pop("tracked_frames", None)
      uploaded_file = None
  
  # ------------------- VIDEO CONTROLS --------------------------
  def get_obj_by_current_frame():
      outs = []
      for obj in st.session_state.objects:
          if obj["frame_idx"] == st.session_state.frame_idx:
              outs.append(obj)
      return outs

  def render_mask(image: np.ndarray, mask: np.ndarray, color, alpha=0.5):
      colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
      colored_mask = np.moveaxis(colored_mask, 0, -1)
      masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
      image_overlay = masked.filled()
      
      image = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
      return image


  if st.session_state.video_frames:
      num_frames = len(st.session_state.video_frames)

      col1, col2, _ = st.columns([1, 1, 8])
      with col1:
          if st.button("Previous") and st.session_state.frame_idx > 0:
              st.session_state.frame_idx -= 1
      with col2:
          if st.button("Next") and st.session_state.frame_idx < num_frames - 1:
              st.session_state.frame_idx += 1
      # Always get updated frame after idx change
      frame = st.session_state.video_frames[st.session_state.frame_idx]
      image_height, image_width = frame.shape[:2]
      st.subheader(f"🖼️ Frame {st.session_state.frame_idx+1}/{num_frames}")

  # ------------------- CANVAS DRAWING --------------------------
  def get_annotated_image(base_frame, objs: list):
      np_image = base_frame.copy()
      for obj in objs:
          # Draw mask if exists
          if "mask" in obj and obj["mask"] is not None:
              mask = obj["mask"]
              color = st.session_state.obj_to_color[obj["obj_id"]]  # (R, G, B)
              #print(f"{obj['obj_id']}, color {color}")
              np_image = render_mask(np_image, mask, color, 0.7)
              
              
      return Image.fromarray(np_image)

  if st.session_state.video_frames:
      # Toggle label mode
      st.radio("Select label for point:", ["positive", "negative"], key="current_mode", horizontal=True,
              help="`positive` -> objectness meanwhile `negative` -> background")

      if st.button("1. Create New Object"):
          st.session_state.objects.append({
              "points": [],
              "labels": [],
              "mask": None,
              "frame_idx": st.session_state.frame_idx,
              "obj_id": st.session_state.obj_id
          })
          st.session_state.obj_to_color.update(
              {st.session_state.obj_id: tuple(np.random.randint(0, 255, 3).tolist())}
          )
          st.session_state.obj_id += 1
          
      # Get frame and overlay points if available
      if st.session_state.objects and st.session_state.objects[-1]["frame_idx"] == st.session_state.frame_idx:
          last_obj = st.session_state.objects[-1]
          current_objects = get_obj_by_current_frame()
          st.session_state.canvas_img = get_annotated_image(frame, current_objects).convert("RGB")
      else:
          st.session_state.canvas_img = Image.fromarray(frame.copy())

      fill_color = "rgba(0, 255, 0, 0.3)" if st.session_state.current_mode == "positive" else "rgba(255, 0, 0, 0.3)"
      mid_col1, mid_col2 = st.columns(2)
      with mid_col1:
          canvas_result = st_canvas(
              fill_color=fill_color,
              stroke_width=1,
              background_image=st.session_state.canvas_img,
              update_streamlit=True,
              height=frame.shape[0],
              width=frame.shape[1],
              drawing_mode="point",
              key="canvas",
              point_display_radius=10
          )

      # Add point to latest object
      if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
          new_obj = canvas_result.json_data["objects"][-1]
          x, y = new_obj["left"], new_obj["top"]
          if st.session_state.objects:
              last_obj = st.session_state.objects[-1]
              #print(last_obj)
              x = round(x/image_width, 3)
              y = round(y/image_height, 3)
              if last_obj["frame_idx"] == st.session_state.frame_idx and ([x, y] not in last_obj["points"] ):
                  if len(st.session_state.objects) > 1:
                      second_last_obj = st.session_state.objects[-2]
                      if [x, y] not in second_last_obj["points"]:
                          last_obj["points"].append([x, y])
                          last_obj["labels"].append(1 if st.session_state.current_mode == "positive" else 0)
                          last_obj["frame_idx"] = st.session_state.frame_idx
                  else:
                      last_obj["points"].append([x, y])
                      last_obj["labels"].append(1 if st.session_state.current_mode == "positive" else 0)
                      last_obj["frame_idx"] = st.session_state.frame_idx

      # Reset points
      if st.session_state.objects:
          if st.button("Reset Points for Current Object"):
              last_obj = st.session_state.objects[-1]
              if last_obj["frame_idx"] == st.session_state.frame_idx:
                  last_obj["points"] = []
                  last_obj["labels"] = []
                  
          if st.button("Reset all objects"):
            st.session_state.objects = []
            st.session_state.obj_id = 0
            st.session_state["tracked_frames"] = []

      # Display object data
      with mid_col2:
          st.subheader("🧷 Annotated Objects")
          list_input_dict = []
          for each in st.session_state.objects:
              list_input_dict.append({k: v for k, v in each.items() if k != "mask"})
          st.write(list_input_dict)
          #for obj in st.session_state.objects:
          #    st.markdown(f"**Object ID {obj['obj_id']} on Frame {obj['frame_idx']}**")
          #    st.text(f"Points: {obj['points']}")
          #    st.text(f"Labels: {obj['labels']}")

      # Create Mask API call
      if st.button("Create Mask"):
          current_objects = get_obj_by_current_frame()
          for obj in current_objects:
              with st.spinner("Getting mask for current frame"):
                  print(object_to_region(obj)[0].proto)
                  masks = model.predict(
                      image=dt.Image.from_pil(Image.fromarray(frame.copy())),
                      regions=object_to_region(obj),
                      multimask_output=False
                  )
                  # masks = model.predict(
                  #     image=dt.Image.from_pil(Image.fromarray(frame.copy())),
                  #     dict_inputs=dict(
                  #         points=obj["points"],
                  #         labels=obj["labels"]
                  #         ),
                  #     multimask_output=False
                  # )
              mask_bytes = masks[0].proto.region_info.mask.image.base64
              mask = Image.open(io.BytesIO(mask_bytes))
              mask = np.asarray(mask, dtype=np.uint8)
              obj["mask"] = mask
          #st.session_state.canvas_img = get_annotated_image(frame, current_objects).convert("RGB")
          st.rerun()
                  

      # Submit for full tracking
      if st.button("Submit to Track") and uploaded_file:
          list_input_dict = []
          for each in st.session_state.objects:
              list_input_dict.append({k: v for k, v in each.items() if k != "mask"})
          cl_video = dt.Video(bytes=uploaded_file.read())
          with st.expander("View request"):
            st.markdown(f"```model.generate(video={cl_video.__repr__()}, list_dict_inputs={list_input_dict}))```")
          
          tracked_frames: Iterator[dt.Frame] = model.generate(video=cl_video, list_dict_inputs=list_input_dict)
          #tracked_frames: Iterator[dt.Frame] = model.generate(video=cl_video, frames=objects_to_frames(list_input_dict))
          
          st.session_state["tracked_frames"] = []
          count = 0
          view_image = st.empty()
          start_track_time = time.perf_counter()
          total_track_time = []
          with st.spinner("Tracking.."):
              for trk_frame in tracked_frames:
                  _frame = st.session_state.video_frames[count].copy()
                  this_track_time = time.perf_counter() - start_track_time
                  total_track_time.append(this_track_time)
                  for reg in trk_frame.regions:
                      mask_bytes = reg.proto.region_info.mask.image.base64
                      mask = Image.open(io.BytesIO(mask_bytes))
                      mask = np.asarray(mask, dtype=np.uint8)
                      track_id = int(reg.proto.track_id)
                      color = st.session_state.obj_to_color[track_id] 
                      #for obj in st.session_state.objects:
                      _frame = render_mask(_frame, mask, color, 0.7)
                  st.session_state.fps = round((count + 1) / sum(total_track_time), 3)
                  view_image.image(_frame, caption=f"Tracked Frame {count+1}. FPS = {st.session_state.fps} f/s. Time = {round(sum(total_track_time), 3)} sec", channels="RGB")
                  st.session_state["tracked_frames"].append(_frame)
                  count += 1
                  start_track_time = time.perf_counter()
              #view_image.markdown("Done")
      
  if st.session_state.get("tracked_frames"):
      st.subheader("View tracked objects")
      st.markdown(f"Tracking speed: {st.session_state.fps} frame/sec")
      def write_video_from_frames(frames, fps=10):
          height, width, _ = frames[0].shape
          temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", prefix="out-sam2-")
          out = cv2.VideoWriter(temp_file.name, cv2.VideoWriter_fourcc(*'H264'), fps, (width, height))
          for frame in frames:
              bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
              bgr_frame = cv2.resize(bgr_frame, (width, height))
              out.write(bgr_frame)
          out.release()
          
          return temp_file.name
      tracked_video_name = write_video_from_frames(st.session_state.get("tracked_frames"), 30)
      print(tracked_video_name)
      with open(tracked_video_name, "rb") as f:
          video_bytes = f.read()
          print(f"len output video {len(video_bytes)}")
          st.video(video_bytes, format="video/mp4")
      st.download_button(
              label="📥 Download Video",
              data=video_bytes,
              file_name="processed_video.mp4",
              mime="video/mp4"
          )
      os.remove(tracked_video_name)
    

if __name__ == "__main__":
  display()