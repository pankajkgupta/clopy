import cv2
import math

class Rect:
    x = None
    y = None
    w = 6
    h = 6
    # Drag in progress
    drag = False
    # already present
    active = True
    # Marker flags by positions
    hold = False
    name = ''
    color = None
    avgDff = 0
    def __init__(self, name, x=None, y=None, w=None, h=None, color = None):
        self.name = name
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.active = True
        
    def area(self):
        return self.w * self.h
    
    def printit(self):
        print(str(self.x) + ',' + str(self.y) + ',' + str(self.w) + ',' + str(self.h))
        
class Circle:
    x = None
    y = None
    r = 6
    # Drag in progress
    drag = False
    # already present
    active = True
    # Marker flags by positions
    hold = False
    name = ''
    color = None

    def __init__(self, name, x=None, y=None, r=None, color=None):
        self.name = name
        self.x = x
        self.y = y
        self.r = r
        self.color = color
        self.active = True
    
    def area(self):
        return math.pi * (self.r**2)
    
    def printit(self):
        print(str(self.x) + ',' + str(self.y) + ',' + str(self.r))

# endclass

class annots:
    # Limits on the canvas
    keepWithin = Rect('mainwin')

    # To store rectangle anchor point
    # Here the rect class object is used to store
    # the distance in the x and y direction from
    # the anchor point to the top-left and the bottom-right corner
    selectedJoint = None
    # Selection marker size
    sBlk = 2
    # Whether initialized or not
    initialized = False

    rois = {}

    # Image
    image = None

    # Window Name
    wname = ""

    multiframe = 0
    # Return flag
    returnflag = False
    frame_n = 0
    colorDict = {}
    def __init__(self, name):
        self.name = name
        # To store circle
        self.rois[name] = Rect(name)

# endclass


def init(annot_obj, rois, windowName, windowWidth, windowHeight):
    # Image
    # annot_obj.image = Img

    # Window name
    annot_obj.wname = windowName
    # Limit the selection box to the canvas
    annot_obj.keepWithin.x = 0
    annot_obj.keepWithin.y = 0
    annot_obj.keepWithin.w = windowWidth
    annot_obj.keepWithin.h = windowHeight

    frame_n = 0
    annot_obj.rois = rois       

# enddef

def dragcircle(event, x, y, flags, dragObj):
    # if x < dragObj.keepWithin.x:
    #     x = dragObj.keepWithin.x
    # # endif
    # if y < dragObj.keepWithin.y:
    #     y = dragObj.keepWithin.y
    # # endif
    # if x > (dragObj.keepWithin.x + dragObj.keepWithin.w - 1):
    #     x = dragObj.keepWithin.x + dragObj.keepWithin.w - 1
    # # endif
    # if y > (dragObj.keepWithin.y + dragObj.keepWithin.h - 1):
    #     y = dragObj.keepWithin.y + dragObj.keepWithin.h - 1
    # endif


    if event == cv2.EVENT_LBUTTONDOWN:
        mouseDown(x, y, dragObj)
    # endif
    if event == cv2.EVENT_LBUTTONUP:
        mouseUp(x, y, dragObj)
    # endif
    if event == cv2.EVENT_MOUSEMOVE:
        mouseMove(x, y, dragObj)
    # endif
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouseDoubleClick(x, y, dragObj)
    # endif

# enddef

def pointInRect(pX, pY, rX, rY, rW, rH):
    if rX <= pX <= (rX + rW) and rY <= pY <= (rY + rH):
        return True
    else:
        return False
    # endelseif


def pointInCircle(pX, pY, rX, rY, rR):
    if ((pX - rX)**2 ) + ((pY - rY)**2) < rR**2:
        return True
    else:
        return False
    # endelseif

# enddef

def updateAnnots(annots_obj, frame_n, im):

    rois = annots_obj.rois.keys()
    if annot_df.empty:
        return
    # This has to be below all of the other conditions
    # if pointInCircle(eX, eY, dragObj.outCircle.x, dragObj.outCircle.y, dragObj.outCircle.r):
    annots_obj.image = im
    annots_obj.frame_n = frame_n
    for joint in annot_df:
        annots_obj.rois[joint].x, annots_obj.rois[joint].y, _score = [round(float(i)) for i in annot_df[joint].values[0].split('-')]

    #clearCanvasNDraw(annots_obj)
    return

def mouseDoubleClick(eX, eY, dragObj):

    # if pointInCircle(eX, eY, dragObj.outCircle.x, dragObj.outCircle.y, dragObj.outCircle.r):
    dragObj.returnflag = True
    cv2.destroyWindow(dragObj.wname)
    # endif
# enddef

def mouseDown(x, y, dragObj):

    if dragObj.selectedJoint:

        return

    else:
        for roi in dragObj.rois:

            if roi.x == 0:
                continue

            if pointInRect(x, y, int(roi.x), int(roi.y), int(roi.w), int(roi.h)):
                dragObj.selectedJoint = roi
                dragObj.selectedJoint.x = x
                dragObj.selectedJoint.y = y
                dragObj.selectedJoint.drag = True
                dragObj.selectedJoint.active = True
                dragObj.selectedJoint.hold = True

# enddef

def mouseMove(eX, eY, dragObj):

    if dragObj.selectedJoint:

        jt = dragObj.selectedJoint
        # jt.x = eX - dragObj.anchor[jt.name].x
        # jt.y = eY - dragObj.anchor[jt.name].y
        jt.x = eX
        jt.y = eY

        if jt.x < dragObj.keepWithin.x:
            jt.x = dragObj.keepWithin.x
        # endif
        if jt.y < dragObj.keepWithin.y:
            jt.y = dragObj.keepWithin.y
        # endif
        if (jt.x + jt.w) > (dragObj.keepWithin.x + dragObj.keepWithin.w - 1):
            jt.x = dragObj.keepWithin.x + dragObj.keepWithin.w - 1 - jt.w
        # endif
        if (jt.y + jt.h) > (dragObj.keepWithin.y + dragObj.keepWithin.h - 1):
            jt.y = dragObj.keepWithin.y + dragObj.keepWithin.h - 1 - jt.h
        # endif

        # update the joint with score 10 since this is done by a human annotator
        #if dragObj.multiframe:
        #    dragObj.joints_df.loc[dragObj.joints_df['frame_n'] >= dragObj.frame_n, jt.name] = str(jt.x) + '-' + str(
        #        jt.y) + '-10'
        #else:
        #    dragObj.joints_df.loc[dragObj.joints_df['frame_n'] == dragObj.frame_n, jt.name] = str(jt.x) + '-' + str(jt.y) + '-10'
        #clearCanvasNDraw(dragObj)
        return
    # endif


# enddef

def mouseUp(eX, eY, dragObj):
    if dragObj.selectedJoint:
        dragObj.selectedJoint.drag = False
        disableResizeButtons(dragObj)
        dragObj.selectedJoint.hold = False
        dragObj.selectedJoint.active =False
        dragObj.selectedJoint = None

        # endif

        #clearCanvasNDraw(dragObj)

# enddef

def disableResizeButtons(dragObj):
    dragObj.hold = False


# enddef

def clearCanvasNDraw(dragObj):
    # Draw
    tmp = dragObj.image.copy()
    tmp1 = dragObj.image.copy()
    for joint_name in dragObj.rois:
        joint = dragObj.rois[joint_name]
        if joint.x == 0:
            return
        cv2.rectangle(tmp, (joint.x, joint.y),
                  (joint.x + joint.w,
                   joint.y + joint.h), dragObj.colorDict[joint_name], 2)
    cv2.imshow(dragObj.wname, tmp)
    # cv2.waitKey()


# enddef
