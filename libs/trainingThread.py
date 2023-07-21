from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
import os

class TrainingThread(QThread):
	finished = pyqtSignal(int)
	progress = pyqtSignal(int)
	export_model = pyqtSignal(str)
	model_map = pyqtSignal(int)
	
	def __init__(self, num_epochs, model_path):
		QThread.__init__(self)
		self.num_epochs = num_epochs
		self.model_path = model_path
		self.process = 0

	def run(self):
		model = YOLO('yolov8n.pt')
		#model = YOLO('yolov8n.yaml')
		print(model.info())
		self.process = 0
		model.add_callback("on_train_epoch_start", self.on_train_epoch_start)
		model.add_callback("on_train_epoch_end", self.on_train_epoch_end)
		model.add_callback("on_train_end", self.on_train_end)
		model.add_callback("teardown", self.on_train_teardown)	

		model.train(data=os.path.join(self.model_path, "yolov8al.yaml"), imgsz=640, epochs=self.num_epochs,
			batch=8, name='yolov8n_al', project=self.model_path, exist_ok=True) #resume=True
		results = model.val()
		print("Val results:", results)
		path = model.export(format="onnx")
		print("Model path:", path, "map50:", results.box.map50)
		self.export_model.emit(path)
		self.model_map.emit(results.box.map50)
		#results = model.predict(source='https://media.roboflow.com/notebooks/examples/dog.jpeg', conf=0.25)		
		
	def on_train_end(self, trainer):
		self.process = self.process + 1
		self.finished.emit(self.process)

	def on_train_epoch_start(self, trainer):
		print("Starting new epoch")

	def on_train_epoch_end(self, trainer):
		self.process = self.process + 1
		self.progress.emit(self.process)

	def on_train_teardown(self, trainer):
		self.process = self.process + 1
		self.progress.emit(self.process)			