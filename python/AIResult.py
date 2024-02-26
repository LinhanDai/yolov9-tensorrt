class DetResult(object):
    def __init__(self, score, box, class_id):
        self.score = score      # confidence
        self.box = box          # x1,y1,w,h
        self.class_id = class_id    # class_id