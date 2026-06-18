import PIL.Image
from kraken.lib import vgsl
from kraken.serialization import serialize as serialize
from kraken import blla
from kraken.lib import models
from kraken import rpred
import kraken.containers

from typing import Union, Self
from dataclasses import dataclass
import dataclasses

@dataclass
class OCRLine:
	"""
	Classe principale pour la description d'une ligne en sortie de Kraken ou Party. Contient la baseline, la prédiction,
	les cuts (= les polygones individuels pour chaque caractère prédit) et le polygone le cas échéant
	"""
	image_path: str
	baseline: list
	prediction: str
	cuts: list
	polygon: list | None = None
	prediction_with_deletion: str | None = None


class OCRRecord():
	"""
	Classe principale qui contient les résultats de l'OCR.
	Une liste d'objets OCRLine qui contiennent chacun la baseline,
	la prédiction, les cuts, les polygones et le chemin vers l'image.
	"""

	def __init__(self, record: list[dict] = []):
		self.record: list = []
		for line in record:
			Line: OCRLine = OCRLine(baseline=line['baseline'],
									prediction=line['prediction'],
									prediction_with_deletion=None,
									cuts=line['cuts'],
									polygon=line['polygon'],
									image_path=line['image_path'])
			self.record.append(Line)

	def recreate_record(self, list_of_lines: list[OCRLine]):
		self.record: list = []
		for line in list_of_lines:
			self.record.append(line)

	def join_transcription(self, merge_newlines=True) -> str:
		"""
		Cette fonction retourne le texte fusionné de toutes les lignes d'un OCRRecord.
		:return: Une chaîne de caractères.
		"""
		if merge_newlines is False:
			delimiter = "\n"
		else:
			delimiter = " "
		return delimiter.join([line.prediction for line in self.record])

	def index(self, item):
		return self.record.index(item)

	def __iter__(self):
		return iter(self.record)

	def __len__(self) -> int:
		return len(self.record)

	def __getitem__(self, index: int) -> Union["OCRLine", Self]:
		current_slice = self.record[index]
		if isinstance(current_slice, list):
			new_record = OCRRecord()
			new_record.recreate_record(list_of_lines=current_slice)
		else:
			new_record = current_slice
		return new_record

	def __str__(self, show_cuts=False):
		out_dict = []
		for line in self.record:
			if show_cuts is True:
				out_dict.append({"baseline": line.baseline,
								 "cuts": line.cuts,
								 "prediction": line.prediction,
								 "prediction_with_deletion": line.prediction_with_deletion,
								"image_path": line.image_path})
			else:
				out_dict.append({"baseline": line.baseline,
								 "prediction": line.prediction,
								 "polygon": line.polygon,
								 "prediction_with_deletion": line.prediction_with_deletion,
								"image_path": line.image_path})
		return json.dumps(out_dict)

	def to_json(self):
		"""
		Transforme un objet OCRRecord en dictionnaire.
		:return:
		"""
		dictionnary = []
		for line in self.record:
			dictionnary.append({"prediction": line.prediction,
								"baseline": line.baseline,
								"cuts": line.cuts,
								"polygon": line.polygon,
								 "prediction_with_deletion": line.prediction_with_deletion,
								"image_path": line.image_path})
		return dictionnary

	def from_json(self, path) -> None:
		"""
		Instancie un objet de classe OCRRecord à partir d'un fichier JSON.
		:param path: le chemin vers le fichier
		"""
		lines_as_dict = load_json_to_dict(path)
		self.record = []
		for item in lines_as_dict:
			if "polygon" not in item:
				item["polygon"] = None
			try:
				prediction_with_deletion = item['prediction_with_deletion']
			except KeyError:
				prediction_with_deletion = None
			Line: OCRLine = OCRLine(baseline=item['baseline'],
									prediction=item['prediction'],
									cuts=item['cuts'],
									polygon=item['polygon'],
									prediction_with_deletion=prediction_with_deletion,
									image_path=item["image_path"])
			self.record.append(Line)



class KRAKEN():
	"""
	Classe permettant la segmentation en ligne et l'OCR d'une page.
	La segmentation **en zones** n'est pas gérée par cette classe mais pas une classe YOLO.
	"""
	def __init__(self, segmentation_model, ocr_model, device):
		self.segmentation_model = segmentation_model
		self.ocr_model = ocr_model
		self.device = device

	def segment_lines_with_kraken(self, image):
		seg_model = vgsl.TorchVGSLModel.load_model(self.segmentation_model)
		baseline_seg:kraken.containers.Segmentation = blla.segment(image, model=seg_model, device=self.device)
		return baseline_seg



	def serialize(self, prediction):
		"""
		Applique la sérialisation d'un fichier en ALTO.
		:param prediction:
		:return:
		"""
		serialized = serialize(results=prediction,
							   template="alto",
							   sub_line_segmentation=False)
		return serialized


	def predict_with_kraken(self, im:PIL.Image.Image,
							segments:kraken.blla.Segmentation,
							extract_polygons:bool = False,
							return_kraken_preds = False,
							image_name = None) -> OCRRecord:
		"""
		Production de l'inférence à l'aide d'un modèle kraken et de segments.
		:param im: L'image chargée
		:param segments: Les segments (objet Kraken)
		:return: un objet OCRRecord.
		"""
		model = models.load_any(self.ocr_model, device=self.device)
		pred_it = rpred.rpred(model, im, segments)
		if return_kraken_preds == True:
			results = dataclasses.replace(pred_it.bounds, lines=[item for item in pred_it], imagename=image_name)
			return results
		prediction = []
		for line, record in zip(segments.lines, pred_it):
			interm_dict = {}
			interm_dict['baseline'] = line.baseline
			if extract_polygons:
				interm_dict['polygon'] = line.boundary
			else:
				interm_dict['polygon'] = None
			interm_dict['prediction'] = record.prediction
			interm_dict['cuts'] = record.cuts
			interm_dict['image_path'] = image_name
			prediction.append(interm_dict)
		my_OCR_record = OCRRecord(record=prediction)
		return my_OCR_record
