from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
import os

class TrainingThread(QThread):
	finished = pyqtSignal(int)
	progress = pyqtSignal(int)
	
	def __init__(self, num_epochs, model_path):
		QThread.__init__(self)
		self.num_epochs = num_epochs
		self.model_path = model_path
		self.process = 0

	def run(self):
		model = YOLO('yolov8n.pt')
		self.process = 0
		model.add_callback("on_train_epoch_start", self.on_train_epoch_start)
		model.add_callback("on_train_epoch_end", self.on_train_epoch_end)
		model.add_callback("on_train_end", self.on_train_end)
		model.add_callback("teardown", self.on_train_teardown)	

		results = model.train(data=os.path.join(self.model_path, "yolov8al.yaml"), imgsz=640, epochs=self.num_epochs, batch=8, name='yolov8n_al') #resume=True
		print("Training results:", results)
		#results = model.val()
		success = model.export(format='onnx')
		print("Export success:", success)
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