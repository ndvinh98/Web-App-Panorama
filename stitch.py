import cv2
import numpy as np
import features
from imutils import paths

def loadImages(path,resize):
    '''Load Images from @path to list, if @resize=true, image will be resized'''
    image_path = list(paths.list_images(path))
    list_image = []
    for i,j in enumerate(image_path):
        image = cv2.imread(j)
        if resize==1:
            image=cv2.resize(image,(int(image.shape[1]/4),int(image.shape[0]/4)))
        list_image.append(image)
    return (list_image)

def warpTwoImages(src_img, dst_img,showstep=False,option='ORB',ratio=0.75):
    '''warp 2 images'''
	#generate Homography matrix
    H,_=features.generateHomography(src_img,dst_img,option=option,ratio=ratio)

	#get height and width of two images
    height_src,width_src = src_img.shape[:2]
    height_dst,width_dst = dst_img.shape[:2]

	#extract conners of two images: top-left, bottom-left, bottom-right, top-right
    pts1 = np.float32([[0,0],[0,height_src],[width_src,height_src],[width_src,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,height_dst],[width_dst,height_dst],[width_dst,0]]).reshape(-1,1,2)
    
    try:
        #aply homography to conners of src_img
        pts1_ = cv2.perspectiveTransform(pts1, H)
        pts = np.concatenate((pts1_, pts2), axis=0)

        #find max min of x,y coordinate
        [xmin, ymin] = np.int64(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int64(pts.max(axis=0).ravel() + 0.5)
        t = [-xmin,-ymin]

    
        #top left point of image which apply homography matrix, which has x coordinate < 0, has side=left
        #otherwise side=right (stich to right side of dst_img)
        if(pts[0][0][0]<0): 
            side='left'
            width_pano=width_dst+t[0]
        else:
            width_pano=int(pts1_[3][0][0])
            side='right'
        height_pano=ymax-ymin

        #Translation 
        #https://stackoverflow.com/a/20355545
        Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) 
        src_img_warped = cv2.warpPerspective(src_img, Ht.dot(H), (width_pano,height_pano))

        #resize dst_img to the same size as src_img_warped
        dst_img_rz=np.zeros((height_pano,width_pano,3))
        if side=='left':
            dst_img_rz[t[1]:height_src+t[1],t[0]:width_dst+t[0]] = dst_img
        else:
            dst_img_rz[t[1]:height_src+t[1],:width_dst] = dst_img

        #blending panorama, if @display=true, function will return left-side, right-side, of panorama and panorama w/o blending
        pano,nonblend,leftside,rightside=panoramaBlending(dst_img_rz,src_img_warped,width_dst,side,showstep=showstep)

        #croping black region
        pano=crop(pano,height_dst,pts)
        return pano,nonblend,leftside,rightside
    except:
        raise Exception("Please try again with another image set Or swich slow method  or fast method!")

def multiStitching(list_images,option='ORB',ratio=0.75):
    '''Choose middle image then divide the array into 2 sub-arrays, left-array and right-array. 
    Stiching middle image with each image in 2 sub-arrays. 
    '''
    if(len(list_images)==2):
        fullpano,_,_,_=warpTwoImages(list_images[0],list_images[1])
    elif (len(list_images)>2):
        n=int(len(list_images)/2+0.5)
        left=list_images[:n]
        right=list_images[n-1:]
        right.reverse()
        while len(left)>1:
            dst_img=left.pop()
            src_img=left.pop()
            left_pano,_,_,_=warpTwoImages(src_img,dst_img,option=option,ratio=ratio)
            left_pano=left_pano.astype('uint8')
            left.append(left_pano)

        while len(right)>1:
            dst_img=right.pop()
            src_img=right.pop()
            right_pano,_,_,_=warpTwoImages(src_img,dst_img,option=option,ratio=ratio)
            right_pano=right_pano.astype('uint8')
            right.append(right_pano)

        #if width_right_pano > width_left_pano, Select right_pano as destination. Otherwise is left_pano
        if(right_pano.shape[1]>=left_pano.shape[1]):
            fullpano,_,_,_=warpTwoImages(left_pano,right_pano,option=option,ratio=ratio)
        else:
            fullpano,_,_,_=warpTwoImages(right_pano,left_pano,option=option,ratio=ratio)
    else:
        raise Exception("Select at least 2 photos to create panorama ")
    return fullpano

def blendingMask(height, width, barrier, smoothing_window, left_biased=True):
    '''create alpha mask.
       @param barrier is x-coordinates of Boundary line between two photos.
       @param smoothing_window is the width of the intersection of two photos.
       @param left_biased=True ->> create left mask, otherwise create right mask
    '''
    assert barrier < width
    mask = np.zeros((height, width))
    
    offset = int(smoothing_window/2)
    try:
        if left_biased:
            mask[:,barrier-offset:barrier+offset+1]=np.tile(np.linspace(1,0,2*offset+1).T, (height, 1))
            mask[:,:barrier-offset] = 1
        else:
            mask[:,barrier-offset:barrier+offset+1]=np.tile(np.linspace(0,1,2*offset+1).T, (height, 1))
            mask[:,barrier+offset:] = 1
    except:
        if left_biased:
            mask[:,barrier-offset:barrier+offset+1]=np.tile(np.linspace(1,0,2*offset).T, (height, 1))
            mask[:,:barrier-offset] = 1
        else:
            mask[:,barrier-offset:barrier+offset+1]=np.tile(np.linspace(0,1,2*offset).T, (height, 1))
            mask[:,barrier+offset:] = 1
    
    return cv2.merge([mask, mask, mask])
    
def panoramaBlending(dst_img_rz,src_img_warped,width_dst,side,showstep=False):
    '''
    create panorama by adding 2 matrices @dst_img_rz and @src_img_warped together then blending
    @width_dst is the width of dst_img before resize.
    @side is the direction of src_img_warped
    '''

    h,w,_=dst_img_rz.shape
    smoothing_window=int(width_dst/8)
    barrier = width_dst -int(smoothing_window/2)
    mask1 = blendingMask(h, w, barrier, smoothing_window = smoothing_window, left_biased = True)
    mask2 = blendingMask(h, w, barrier, smoothing_window = smoothing_window, left_biased = False)

    if showstep:
        nonblend=src_img_warped+dst_img_rz
    else:
        nonblend=None
        leftside=None
        rightside=None

    if side=='left':
        dst_img_rz=cv2.flip(dst_img_rz,1)
        src_img_warped=cv2.flip(src_img_warped,1)
        dst_img_rz=(dst_img_rz*mask1)
        src_img_warped=(src_img_warped*mask2)
        pano=src_img_warped+dst_img_rz
        pano=cv2.flip(pano,1)
        if showstep:
            leftside=cv2.flip(src_img_warped,1)
            rightside=cv2.flip(dst_img_rz,1)
    else:
        dst_img_rz=(dst_img_rz*mask1)
        src_img_warped=(src_img_warped*mask2)
        pano=src_img_warped+dst_img_rz
        if showstep:
            leftside=dst_img_rz
            rightside=src_img_warped

    
    return pano,nonblend,leftside,rightside

def crop(panorama,h_dst,conners):
    '''crop panorama based on destination image (dst_img).
    @param panorama is the panorama
    @param h_dst is the hight of destination image
    @param conner is the tuple which containing 4 conners of warped image and 
    4 conners of destination image'''
    #find max min of x,y coordinate
    [xmin, ymin] = np.int32(conners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(conners.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    conners=conners.astype(int)

   #top-left<0 ->>> side = left, otherwise side=right
    if conners[0][0][0]<0:
        n=abs(-conners[1][0][0]+conners[0][0][0])
        panorama=panorama[t[1]:h_dst+t[1],n:,:]
    else:
        if(conners[2][0][0]<conners[3][0][0]):
            panorama=panorama[t[1]:h_dst+t[1],0:conners[2][0][0],:]
        else:
            panorama=panorama[t[1]:h_dst+t[1],0:conners[3][0][0],:]
    return panorama



