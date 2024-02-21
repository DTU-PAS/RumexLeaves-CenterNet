from multiprocessing import Lock
from annotation_converter.AnnotationConverter import AnnotationConverter

# Thread-safe variant of the Annotation Converter).
class AnnotationConverterTT:
    def __init__(self):
        self._attr_lock = Lock()
        self._file_locks = {}

    def write_cvat(self, annotations, annotation_file):
        self._add_lock(annotation_file)
        self._file_locks[annotation_file].acquire()
        AnnotationConverter.write_cvat(annotations, annotation_file)
        self._file_locks[annotation_file].release()

    def _add_lock(self, annotation_file):
        self._attr_lock.acquire()
        if annotation_file not in self._file_locks.keys():
            self._file_locks[annotation_file] = Lock()
        self._attr_lock.release()

    def remove_cvat(self, ann, annotation_file):
        self._add_lock(annotation_file)
        self._file_locks[annotation_file].acquire()
        AnnotationConverter.remove_cvat(ann, annotation_file)
        self._file_locks[annotation_file].release()

    def extend_cvat(self, ann, annotation_file):
        self._add_lock(annotation_file)
        self._file_locks[annotation_file].acquire()
        AnnotationConverter.extend_cvat(ann, annotation_file)
        self._file_locks[annotation_file].release()

    def read_cvat_by_id(self, annotation_file, img_id):
        self._add_lock(annotation_file)
        self._file_locks[annotation_file].acquire()
        annotation = AnnotationConverter.read_cvat_by_id(annotation_file, img_id)
        self._file_locks[annotation_file].release()
        return annotation









