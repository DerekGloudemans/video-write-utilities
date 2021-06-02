import cv2
import cv2 as cv
import numpy as np
import argparse
import skimage.transform
import matplotlib.pyplot as plt





class Fisheye_Rectifier():
    
    def __init__(self,fvp,rip):
        """
        fvp - string, fisheye video path
        rip - string, rectified image path
        """
        
        
        
        #load first frame from video
        # open VideoCapture
        cap = cv2.VideoCapture(fvp)
        ret,frame = cap.read()
        
        self.fisheye_im = frame
        self.bg_im = cv2.imread(rip)
        
        # resize bg_im so it has same heights as fisheye_im
        fish_height = self.fisheye_im.shape[0]
        bg_height = self.bg_im.shape[0]
        
        ratio = fish_height / float(bg_height)
    
        new_size = (int(self.bg_im.shape[1]*ratio),int(self.bg_im.shape[0]*ratio))
        self.bg_im = cv2.resize(self.bg_im,(new_size))
        
        self.distorted_points = []
        self.undistorted_points = []
        
        
        self.start_point = None # used to store click temporarily
        self.clicked = False     
        self.plot()
        
        self.plot_ds = 3
        
        self.colors = np.random.rand(1000,3)*255
    
    def on_mouse(self,event, x, y, flags, params):
    
       if event == cv.EVENT_LBUTTONDOWN and not self.clicked:
         self.start_point = np.array([x,y])
         self.clicked = True
         
       elif event == cv.EVENT_LBUTTONUP:
            self.clicked = False
             
            self.distorted_points.append(self.start_point*self.plot_ds)
            self.undistorted_points.append(np.array([x*self.plot_ds - self.fisheye_im.shape[1],y*self.plot_ds])) 
            self.plot()
                 
    def plot(self,show_lines = True):
        """
        Replots current frame
        """
                
            
        self.cur_frame = np.hstack((self.fisheye_im,self.bg_im))
        
        for i in range(len(self.distorted_points)):

                
                corner_a = self.distorted_points[i][0:2].copy()
                corner_b = self.undistorted_points[i][0:2].copy()
                color = self.colors[i]
                corner_b[0] += self.fisheye_im.shape[1]
                
                self.cur_frame = cv2.line(self.cur_frame,(int(corner_a[0]),int(corner_a[1])),(int(corner_b[0]),int(corner_b[1])),color,2)
                
     
    
        
    def quit(self):
        cv2.destroyAllWindows()
        #self.find_optimal_warp()
        
        #compute K and D
        n = len(self.undistorted_points)
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
        
        undistorted = [np.concatenate((np.array(self.undistorted_points),np.ones([n,1])),axis = 1)[np.newaxis,:,:].astype(np.float32)]
        distorted = [np.array(self.distorted_points)[np.newaxis,:,:].astype(np.float32)]
        
        rvecs = [np.zeros((1, 1, 3), dtype=np.float32) for i in range(n)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float32) for i in range(n)]
        D = np.zeros([4,1]).astype(np.float32)
        K = np.zeros([3,3]).astype(np.float32)
        err, K, D, rvecs, tvecs = cv2.fisheye.calibrate(undistorted,distorted,self.fisheye_im.shape[0:2],K,D,rvecs,tvecs)
        print(err,K,D)
        
        D = np.array([5,1,0,0])
        output_im = cv2.undistort(self.fisheye_im,K,D)
        
        cv2.imshow("frame",output_im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return K,D
        
        
    def find_optimal_warp(self, epsilon = 100,delta = 1e-10,alpha = 0.01,order = 2,affine = True):
        """
        Using gradient descent, find optimal parameters for polynomial expression for x and y coordinate warp function
        
        x' = Ax^2 + By^2 + Cxy + Dx + Ey + F
        y' = Gx^2 + Hy^2 + Ixy + Jx + Ky + L
        
        Use gradient descent to solve for the optimal parameters that minimize the sum of projection error:
        (x' - x'_true)^2 + (y'-y'_true)^2 
        over all input point pairs
        
        """
        points = np.array(self.distorted_points)
        targets = np.array(self.undistorted_points)
        cv2.namedWindow("window")

        
        for order in range(0,5):
        
            if not affine:
                transform = skimage.transform.PolynomialTransform()
                transform.estimate(points,targets,order = order)
            
            else:
                transform = skimage.transform.PiecewiseAffineTransform()
                transform.estimate(points,targets)
            
            tf_points = transform(points)
            error = compute_error(tf_points,targets)
            print("Error sum for polynomial of degree {}: {}".format(order,error))
            
            if not affine:
                inverse_transform = skimage.transform.PolynomialTransform()
                inverse_transform.estimate(targets,points,order = order)
            else:
                inverse_transform = skimage.transform.PiecewiseAffineTransform()
                inverse_transform.estimate(targets,points)
            
            
            transformed_image = skimage.transform.warp(self.fisheye_im,inverse_transform)
            bg = (self.bg_im[:transformed_image.shape[0],:transformed_image.shape[1],:]).astype(float) /256.0
            a = 0.25
            transformed_image = cv2.addWeighted(bg,a,transformed_image,1-a,0)
            transformed_image = cv2.resize(transformed_image,(1920,1080))
            
            cv2.imshow("window",transformed_image)
            title = "Polynomial order {}".format(order)
            cv2.setWindowTitle("window",str(title))
            cv2.waitKey(1)
            cv2.imwrite("trasnform_degree{}.png".format(order),transformed_image*255)
               
            
        cv2.destroyAllWindows()
        # params = np.ones(12) * 1e-08
        # params[[3,10]] = 1
        
        # alpha = np.ones(12) * alpha
        # alpha[11] *= 1e+8
        # alpha[5] *= 1e+8
        
        # err = np.inf
        # iteration = 0
        
        # while err > epsilon:
            
            
            
        #     # tweak each param and compute gradient
        #     gradient = compute_gradient(points,targets,params,delta = delta)
            
        #     # step in direction of gradient
        #     params += -gradient * alpha 
            
        #     # periodically, plot transform
        #     if iteration % 100 == 0:
        #         tf_points = transform_points(points,params)
        #         err = compute_error(tf_points,targets)
        #         print("On iteration {}. Current total error: {}".format(iteration,err))
        #     if iteration % 1000 == 0:
        #         transformed_image = skimage.transform.warp(self.fisheye_im,transform_points,map_args = {"params":params})
        #         bg = (self.bg_im[:transformed_image.shape[0],:transformed_image.shape[1],:]).astype(float) /256.0
        #         a = 0.25
        #         transformed_image = cv2.addWeighted(bg,a,transformed_image,1-a,0)
        #         transformed_image = cv2.resize(transformed_image,(1920,1080))
        #         cv2.imshow("frame",transformed_image)
        #         cv2.waitKey(1)
                
        #     iteration += 1

    def undo(self):
        self.distorted_points = self.distorted_points[:-1]
        self.undistorted_points = self.undistorted_points[:-1]
        self.plot()
    
    def run(self):
        """
        Main processing loop
        """
        
        try:
            self.distorted_points = points
            self.undistorted_points = targets
            self.quit()
            #self.find_optimal_warp()
        
        
        except:
            cv2.namedWindow("window")
            cv.setMouseCallback("window", self.on_mouse, 0)
            
            while True: # one frame
               
                 
               cur_frame = self.cur_frame.copy()
               cur_frame = cv2.resize(cur_frame,(cur_frame.shape[1]//self.plot_ds,cur_frame.shape[0]//self.plot_ds))
               cv2.imshow("window", cur_frame)
               title = "Dummy Title"
               cv2.setWindowTitle("window",str(title))
               
               key = cv2.waitKey(1)
               if key == ord("q"):
                    self.quit()
                    break
               elif key == ord("u"):
                    self.undo()
                    
                

def transform_points(points,params):
    output = np.zeros(points.shape)
    
    output[:,0] =     params[0]*np.power(points[:,0],0.5) \
                    + params[1]*np.power(points[:,1],0.5) \
                    + params[2]*points[:,0]*points[:,1] \
                    + params[3]*points[:,0] \
                    + params[4]*points[:,1] \
                    + params[5]
    
    output[:,1] =     params[6]*np.power(points[:,0],0.5) \
                    + params[7]*np.power(points[:,1],0.5) \
                    + params[8]*points[:,0]*points[:,1] \
                    + params[9]*points[:,0] \
                    + params[10]*points[:,1] \
                    + params[11]
                    
    return output

def compute_error(points,targets):
    x_err = np.power((points[:,0] - targets[:,0]),2)
    y_err = np.power((points[:,1] - targets[:,1]),2)
    err = np.sqrt(x_err + y_err) 

    # x_err = np.abs((points[:,0] - targets[:,0]))
    # y_err = np.abs((points[:,1] - targets[:,1]))
    # err = x_err + y_err
    err = np.average(err)
    return err

def compute_gradient(points,targets,params,delta = 1e-05):
    
    tf_points = transform_points(points,params)
            
    # compute error with current params
    error = compute_error(tf_points,targets)
    
    working_params = params.copy()
    
    gradient = np.zeros(params.shape)
    for i in range(len(params)):
        working_params[i] += delta
        tf_points = transform_points(points,working_params)
        working_error = compute_error(tf_points,targets)
        
        gradient[i] = (working_error - error)#/delta
    
    return gradient
    

        
        
    





if __name__ == "__main__":
    
    try:
        raise Exception
        
    except:
        
        points = np.array([[3333,  633],
                            [2988,  558],
                            [2547,  486],
                            [2031,  423],
                            [1503,  387],
                            [1011,  384],
                            [ 606,  399],
                            [ 558,  495],
                            [ 972,  480],
                            [1491,  495],
                            [2043,  531],
                            [2580,  594],
                            [3036,  666],
                            [3384,  732],
                            [3471,  852],
                            [3132,  798],
                            [2673,  729],
                            [2115,  666],
                            [1542,  624],
                            [ 996,  603],
                            [1047, 1200],
                            [1002, 1383],
                            [ 978, 1590],
                            [1737, 1257],
                            [1692, 1452],
                            [1698, 1677],
                            [2442, 1314],
                            [2436, 1500],
                            [2466, 1710],
                            [3036, 1332],
                            [3042, 1503],
                            [3090, 1698],
                            [3459, 1344],
                            [3483, 1494],
                            [3519, 1662],
                            [ 204, 1101],
                            [ 120, 1413]])
        
        targets = np.array([[3342,  426],
                            [2811,  420],
                            [2292,  417],
                            [1773,  408],
                            [1251,  408],
                            [ 723,  399],
                            [ 213,  390],
                            [ 213,  546],
                            [ 738,  546],
                            [1257,  567],
                            [1770,  567],
                            [2301,  576],
                            [2814,  582],
                            [3333,  564],
                            [3414,  732],
                            [2889,  738],
                            [2376,  735],
                            [1851,  723],
                            [1071,  723],
                            [ 543,  705],
                            [1017, 1311],
                            [1020, 1467],
                            [1029, 1629],
                            [1545, 1323],
                            [1539, 1473],
                            [1545, 1632],
                            [2067, 1329],
                            [2064, 1485],
                            [2067, 1647],
                            [2598, 1338],
                            [2586, 1494],
                            [2586, 1653],
                            [3105, 1350],
                            [3108, 1503],
                            [3108, 1659],
                            [  96, 1296],
                            [  96, 1614]])
        
        
        fisheye_video_path = "/home/worklab/Data/cv/video/fisheye/record_32_p2c4_00000.mp4"
        rectified_im_path = "/home/worklab/Data/cv/video/fisheye/FOV_cropped.png"
        
        fish = Fisheye_Rectifier(fisheye_video_path,rectified_im_path)
        fish.run()
        
        
                           