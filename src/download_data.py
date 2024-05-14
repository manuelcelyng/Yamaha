from roboflow import Roboflow

rf = Roboflow(api_key="bq72UM3TJLbUG43y7lcs", model_format="yolov8")
dataset = rf.workspace().project("piezas-jolpt").version(1).download(location="./content/my-datasets")