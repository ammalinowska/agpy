import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import os
from scipy import stats
import trackpy as tp
# import pims
import skimage.io as io
from math import sqrt
from pprint import pprint
import itertools
import sys

class FrameImg:
    ''' Add Description '''
    # ROI_crop = [450, 700, 880, 1200]
    # ROI_crop = [300, 900, 480, 700] 
    ROI_crop = [50, 800, 300, 1250] # Drop left and bottom shit
    # ROI_crop = [500, 800, 350, 600] 
    [ylow, yup, xleft, xright] = ROI_crop
    crop_boo = True

    def __init__(self, file_name, file_path = os.getcwd(), frame_num = 1):
        self.file_name = file_name
        self.file_path = file_path
        self.frame_num = frame_num

        img, img_treshold = self.load_img()
        self.get_img_contours(img_treshold)
        self.process_contours()
        # quit()
        self.create_crystal_attr_list()
        self.check_contours()
        # self.drop_upper_outlier_area_crystal()
        # self.drop_edge_contours()


    def drop_upper_outlier_area_crystal(self):
        self.crystal_areas.sort()
        c_max = self.crystal_areas[len(self.crystal_areas) - 1] # Get largest crystal area
        c_max_s = self.crystal_areas[len(self.crystal_areas) - 2] # Get second largest crystal area
        area_ratio = c_max / c_max_s
        if area_ratio > 5: # If largest area is 10 times that of the second largest area
            max_crys = [c for c in self.crystalobjects if c.area == c_max][0] # Retreive crystal object
            print(f'Dropping max crystal, size is {round(area_ratio,2)} times the second biggest crystal')
            # Remove crystal attributes from list, and the crystal form the crystalobjects list.
            self.crystal_areas.remove(max_crys.area)
            self.crystal_lengths.remove(max_crys.length)
            self.contours_lenghts.remove(max_crys.contour_length)
            # self.crystal_centers.remove(max_crys.center_arr)
            # Currently not removing the center coord of max crys. Might cause issues late on
            self.crystalobjects.remove(max_crys)


    def load_img(self):
        ''' Loads in the image, and crops if it crop_boo is set to True. '''
        img_path = os.path.join(self.file_path, self.file_name)
        self.img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if FrameImg.crop_boo:
            img = self.img[FrameImg.ylow:FrameImg.yup, FrameImg.xleft:FrameImg.xright]
        else:
            img = self.img
        self.img_attributes(img)
        img_denoised = self.denoise_img(img)
        img_treshold = self.tresholding_img(img_denoised)
        # Add in possible plot stages method here before the images go out of scope.
        return img, img_treshold

    def img_attributes(self, img):
        ''' Retreive image height and width and store them as an attribute. Used for checking if the
        contours are flipped '''
        self.img_height , self.img_width = img.shape[:2]

    def denoise_img(self, img):
        ''' Desnoise image. 
            Docs : https://docs.opencv.org/2.4/modules/photo/doc/denoising.html '''
        return cv2.fastNlMeansDenoising(src=img, dst=None, h=10,
        templateWindowSize=9, searchWindowSize=23)

    def tresholding_img(self,img_denoised):
        ''' Treshold image. 
            Docs: https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3 '''
        return cv2.adaptiveThreshold(src=img_denoised, maxValue=255,
                adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                thresholdType=cv2.THRESH_BINARY, blockSize=179, C=1)

    def get_img_contours(self, img_treshold):
        ''' Retreive image contours. ## NEED TO WORK WITH RETR_CCOMP instead of RETR_EXTERNAL TO GET HIERACHY FOR DEALIG WITH INCEPTIONS
            Docs: https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a'''
        self.contours, self.hierarchy = cv2.findContours(img_treshold,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # print(self.hierarchy)
# RETR_EXTERNAL

    def process_contours(self):
        ''' First creates two empty lists to store the crystals and the reminaining/other objects. Next, 
            loops through all the retreives contours: If the len(contour), which is the amount of coordinate
            points is large enough, it creates a CrystalObject class instance. Then, if the 'contour lenght'
            is greater than 2, it adds the CrystalObject to the list. If the conditions are not met, the object
            is stored in the other objects list.   '''
        self.crystalobjects = []
        self.otherobjects = []
        num_contours = len(self.contours)
        print(f'Number of contours found: {num_contours}')
        for i, contour in enumerate(self.contours):
            if len(contour) > 30: # Checks the amount of coordinate points in the contour <- NEEDS TO BE REWORKED 
                print(f'Processing img contours {i}/{num_contours}', end = '\r')
                obj = CrystalObject(contour, contour_num= i, frame_num = self.frame_num)
                if obj.x_center == 0 and obj.y_center == 0:
                    self.otherobjects.append(obj)
                else:
                    self.crystalobjects.append(obj) # <- SEEMS TO NEVER BE THE CASE.........

        print('Contour processing done  .......')
        print(f'Number of contours stored: {len(self.crystalobjects)}')
        print(f'Number of other objects stored: {len(self.otherobjects)}')

    def create_crystal_attr_list(self):
        self.crystal_areas = [i.area for i in self.crystalobjects]
        self.crystal_lengths = [i.length for i in self.crystalobjects]
        self.contours_lenghts = [i.contour_length for i in self.crystalobjects]
        self.crystal_centers = [i.center_arr for i in self.crystalobjects]


    def check_contours(self):
        ''' Function thats checks if the x and y axes have been swapped, and corrects this if so.
            Done by checking if the highest x found in the contour coordinates if higher than the image
            width, or the higest y contour coordinate is higher than the image height. 
            Correction is done for both the smoothed contours array, and the smoothed contours dataframe.
            Whether the action is performed or not is stored in the flipped contour attribute.  '''
        self.y_maximum = max([i.s_contours_df['y'].max() for i in self.crystalobjects])
        self.x_maximum = max([i.s_contours_df['x'].max() for i in self.crystalobjects])
        if self.x_maximum > self.img_width or self.y_maximum > self.img_height:
             self.flipped_contours = True
             print(f'{self.x_maximum} vs {self.img_width}; {self.y_maximum} vs {self.img_height} ')
             for c in self.crystalobjects:
                # c.s_contours[:, 0], c.s_contours[:, 1] = c.s_contours[:, 1], c.s_contours[:, 0].copy()
                c.s_contours_df.rename(columns={'x': 'y', 'y': 'x', 'sx' : 'sy', 'sy' : 'sx'}, inplace=True)
        else:
            self.flipped_contours = False
        print(f'Contours flipped: {self.flipped_contours}')


    def drop_edge_contours(self):
        ''' Funtion to cutoff the crystals that are within a 'cutoff_pct' * the width and height of the img.
            First, creates a list of cutoff coordinate values for both axis, then loops through each crystal countour 
            coorinates to check if the cutoff coordinate values occur in the smoothed contours coordinates.
            If so, it removes the crystal from the crystalobjects list, and adds it to the edge_objects list. ''' 
        self.edge_objects = []
        cutoff_pct = 0.01
        
        cutoff_y_val = cutoff_pct * self.img_height
        y_drop = [self.img_height - i for i in range(0,int(cutoff_y_val))]        
        y_drop.extend(list(range(0,int(cutoff_y_val))))
        
        cutoff_x_val = cutoff_pct * self.img_width
        x_drop = [self.img_width - i for i in range(0,int(cutoff_x_val))]
        x_drop.extend(list(range(0,int(cutoff_x_val))))
        pre_drop_count = len(self.crystalobjects)
        for crystal in reversed(self.crystalobjects):
            for valx, valy in zip(x_drop, y_drop):
                if valx in crystal.s_contours_df.sx.values or \
                        valy in crystal.s_contours_df.sy.values:
                    self.crystalobjects.remove(crystal)
                    self.edge_objects.append(crystal)
                    break  # to stop the loop   
        post_drop_count = len(self.crystalobjects)
        print(f'Edge dropping dropped {pre_drop_count - post_drop_count} crystals')       

    # def plot_stages(self):
        ''' Add Description '''
        # if not self.org_img is None:
        #     images = [self.org_img, self.img, self.img_denoised, self.img_treshold]
        #     for i in range(len(images)):
        #         plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
        #         plt.xticks([]), plt.yticks([])
        #     plt.suptitle(self.file_name, fontsize = 8)
        #     plt.show()

    def plot_contours(self, mark_center = False, mark_number = False,
            save_image = False, file_name = f'contourplot1.png'):
        ''' Plot the contours on the orignal image. '''
        self.contoured_img = self.img
        num_crystal = len(self.crystalobjects)
        print(f'Total number of contours to plot: {num_crystal}')
        for i, crystal in enumerate(self.crystalobjects):
            print(f'Drawing contour of crystal {i} / {num_crystal}', end = '\r')
            # Draw original contours in black
            cv2.drawContours(self.contoured_img, crystal.contour_raw,
                                -1, (0, 0, 0), 1)
            # Draw smoothed countours in white.
            cv2.drawContours(self.contoured_img, [crystal.s_contours.astype(int)],
                                -1, (255, 255, 255), 1)
            if mark_center:
                cv2.circle(self.contoured_img, (crystal.x_center, crystal.y_center), 1,
                    (255, 255, 255), -1)
            if mark_number:
                cv2.putText(self.contoured_img, f'{crystal.contour_num}', (crystal.x_center - 2,
                    crystal.y_center - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            if save_image:
                cv2.imwrite(file_name, self.contoured_img)
        print('Crystal contour drawing done ..................')
        cv2.imshow(f'{self.file_name}', self.contoured_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def plot_s_contours(self, show_plot = False,
            save_image = False, frame_img_name = '',
            file_name = f'contourplot3.png'):
        ''' Plots the smoothed contours for each crystalobject in the frame on the original image.  '''
        fig = plt.figure()
        if FrameImg.crop_boo:
            org_img = mpimg.imread(os.path.join(self.file_path,self.file_name))[FrameImg.ylow:FrameImg.yup, FrameImg.xleft:FrameImg.xright]
        else:
            org_img = mpimg.imread(os.path.join(self.file_path,self.file_name))
        plt.imshow(org_img)
        for crystal in self.crystalobjects:
            plt.plot(crystal.s_contours[...,0], crystal.s_contours[...,1])
            plt.scatter(crystal.center_arr[0],crystal.center_arr[1], s=2)
        fig.suptitle(frame_img_name, fontsize = 8)
        if save_image:
            fig.savefig(f'{file_name}')
        plt.close()


class CrystalObject:
    ''' Add Description '''
    def __init__(self, contour_raw, contour_num, frame_num):
        self.contour_raw = contour_raw
        self.frame_num = frame_num
        self.contour_num = contour_num
        self.contour_length = len(self.contour_raw)

        self.moments = cv2.moments(contour_raw)
        self.get_center_point()
        if self.contour_length > 2:
            self.get_area()
            self.get_length()
            self.set_contours_dataframe()
            # print(self.s_contours_df.x.max())

    def get_center_point(self):
        ''' Using the contour moments, retreives the center x and y coordinates,
            and an array of said coordinates. '''
        if self.moments['m00'] != 0:
            self.x_center = int(self.moments['m10'] / self.moments['m00'])
            self.y_center = int(self.moments['m01'] / self.moments['m00'])
        else:
            self.x_center = 0
            self.y_center = 0
        self.center_arr = np.array([self.x_center, self.y_center])

    def get_area(self):
        ''' Sets the area of the contour '''
        self.area = cv2.contourArea(self.contour_raw)

    def get_length(self):
        ''' Sets the 'length' of the contour ??? '''
        self.length = cv2.arcLength(self.contour_raw, True)

    def calculate_curvature(self, df, smoothing):
        ''' Calculates the curvature, ands its mean, of the of curvature coordinates, and creates the sx and sy
            columns, which are the rounded values of the coordinates which used in the edge crystal removal function.
            This function starts by creating df columns of the first and second derivate, together with the helper
            columns. Next, creates the curvature and mean curvature columns, and THRESH_BINARY drops the created
            helper columns. Finally, it removes the padding rows (previously done by the listacrop function). '''
        for z in ['x', 'y']:
            df[f'd{z}'] = np.gradient(df[f'{z}'])
            df[f'd{z}'] = df[f'd{z}'].rolling(smoothing, center=True).mean()
            df[f'd2{z}'] = np.gradient(df[f'd{z}'])
            df[f'd2{z}'] = df[f'd2{z}'].rolling(smoothing, center=True).mean()
            df[f's{z}'] = df[f'{z}'].round(0)
        df['curvature'] = -df.eval('(dx * d2y - d2x * dy) / (dx * dx + dy * dy) ** 1.5')
        df['curvature'] = df.curvature.rolling(smoothing, center=True).mean().round(2)
        df['mean(curvature)'] = np.mean(abs(df.curvature))
        df = df.drop(['dx','d2x','dy','d2y'], axis=1)
        df = df.dropna()    
        return df        

    def set_contours_dataframe(self, smoothing=10):
        ''' Function to prepare the crystal dataframe. Loads in the contours array, padds them (reason?), and then
            smooths the coordinates using rolling mean. Next, calls the calculate_cruvature function in order to 
            retrieve the curvature and its mean. Adds the contour number, frame, area, and lenght of the contour, 
            and saves the resulting df. Finally, it creates an additional df with the rounded coordinates. ''' 
        contours_raw = np.reshape(self.contour_raw, (self.contour_raw.shape[0],2))
        contours = np.pad(contours_raw, ((25,25), (0,0)), 'wrap')
        df = pd.DataFrame(contours, columns={'x', 'y'})
        df = df.reset_index(drop=True).rolling(smoothing).mean().dropna()
        df = self.calculate_curvature(df, smoothing)
        df['contour_num'] = self.contour_num
        df['frame'] = self.frame_num
        df['area'] = self.area
        df['length'] = self.length
        self.s_contours_df = df
        self.s_contours = self.s_contours_df.drop(['curvature','mean(curvature)', 'sx', 'sy', 'contour_num', 'frame', 'area', 'length'], axis=1).to_numpy()
        

class CrystalRecog:
    def __init__(self, c_obj):
        """ Start off by creating the empty lists (which will be appended with the attributes
        for the specific crystal for each FrameImg. Then, add the attributes for the First 
        Frame to start off with. """
        self.s_contours_dfs = []
        self.raw_contours = []
        self.s_contours = []
        self.areas = []
        self.center_arrays = []
        self.mean_curvatures = []
        self.c_count = 0
        self.frames_used = []
        self.cunt_num = c_obj.contour_num
        self.add_crystalobject(c_obj)
        
    def add_crystalobject(self, c_obj):
        """ Add cryal attributes to the respective class intances. """
        self.s_contours_dfs.append(c_obj.s_contours_df)
        self.raw_contours.append(c_obj.contour_raw)
        self.s_contours.append(c_obj.s_contours)
        self.areas.append(c_obj.area)
        self.center_arrays.append(c_obj.center_arr)
        self.mean_curvatures.append(c_obj.s_contours_df['mean(curvature)'].min())
        self.frames_used.append(str(c_obj.frame_num))

        self.max_y = self.s_contours_dfs[len(self.s_contours_dfs)-1].y.max()
        self.min_y = self.s_contours_dfs[len(self.s_contours_dfs)-1].y.min()
        self.max_x = self.s_contours_dfs[len(self.s_contours_dfs)-1].x.max()
        self.min_x = self.s_contours_dfs[len(self.s_contours_dfs)-1].x.min()
        self.c_count += 1

    def add_dubble_crystalobject(self, c_obj1, c_obj2):
        s_contours_df = pd.concat([c_obj1.s_contours_df,c_obj2.s_contours_df])
        self.s_contours_dfs.append(s_contours_df)
        raw_contours = np.concatenate((c_obj1.contour_raw, c_obj2.contour_raw))
        self.raw_contours.append(raw_contours)

        area = (c_obj1.area + c_obj2.area) * 0.5
        self.areas.append(area)
        x_center = (c_obj1.center_arr[0] + c_obj2.center_arr[0])*0.5
        y_center = (c_obj1.center_arr[1] + c_obj2.center_arr[1])*0.5
        center_arr = np.array([x_center, y_center])
        print(f'{c_obj1.center_arr} + {c_obj2.center_arr} = {center_arr}')
        self.center_arrays.append(center_arr)


# MAKE METHOD OF CrystalTracking?
def euqli_dist(p, q, squared=False):
    # Calculates the euclidean distance, the "ordinary" distance between two
    # points. The standard Euclidean distance can be squared in order to place
    # progressively greater weight on objects that are farther apart. This
    # frequently used in optimization problems in which distances only have
    # to be compared.
    if squared:
        return ((p[0] - q[0]) ** 2) + ((p[1] - q[1]) ** 2)
    else:
        return sqrt(((p[0] - q[0]) ** 2) + ((p[1] - q[1]) ** 2))
# MAKE METHOD OF CrystalTracking?
def closest(cur_pos, positions):
    # FIND THE SOURCE FOR THIS CODE ON STACKOVERFLOW!!
    low_dist = float('inf')
    closest_pos = None
    index = None
    for index, pos in enumerate(positions):
        dist = euqli_dist(cur_pos,pos)
        if dist < low_dist:
            low_dist = dist
            closest_pos = pos
            ret_index = index
    return closest_pos, ret_index, low_dist

def get_img_files_ordered(dir_i):
    """ Function returns a list of all input frames, ordered by their frame number, 
    together with a count of the total number of frames.  """
    img_files = []
    for file in os.listdir(dir_i):
        file_i = {
        'filename' : file,
        'file_num' : int(file.split('frame')[1].split('.')[0])
        }
        img_files.append(file_i )
    ordered_img_files = sorted(img_files, key=lambda k: k['file_num'])
    file_count = len(ordered_img_files)
    return ordered_img_files,file_count

def set_and_check_folder(FOLDER_NAME, create_boo = False):
    """ Function to set up a directory and check if it exists."""
    fol_path = os.path.join(os.getcwd(), FOLDER_NAME)
    if os.path.isdir(fol_path):
        return fol_path
    else:
        if create_boo:
            os.mkdir(fol_path)
            print(f'Dir {FOLDER_NAME} has been created.')
            return fol_path
        else:  
            print(f'Dir {FOLDER_NAME} does not seem to exist. Terminating program.')
            sys.exit()

def create_frame_list(img_files, file_count, imgs_dir,
        output_img_dir, IMAGE_FORMAT, plot_boolean = False):
    """
    Function that loops through the input img files folder, and creates
    and instance of FrameImg for each image found with the correct image format.
    Optionally, plots all found contours on the inputted image.
    """
    frame_list = []
    for f_numerator, file in enumerate(img_files):
        file_name = file['filename']
        if file_name.endswith(IMAGE_FORMAT):
            print('------------------------------------------------------')
            print(f'Processing: {file_name}; #{f_numerator + 1}/{file_count}')
            frame_i = FrameImg(file_name, imgs_dir, f_numerator)
            frame_list.append(frame_i)
            if plot_boolean:
                frame_i.plot_s_contours(show_plot = False, save_image = True, frame_img_name = file_name,
                    file_name = os.path.join(output_img_dir, f'contour_plot_frame_{f_numerator + 1}{IMAGE_FORMAT}'))
        else:
            print(f'{file_name} has a different file format than the expected {IMAGE_FORMAT}.')
    return frame_list

if __name__ == "__main__":
    start_time = time.time()
    # Constants:
    IMAGE_FORMAT = '.png'
    INPUT_FOLDER_NAME = 'InputImgs'
    IMAGE_OUTPUT_FOLDER_NAME = 'FR_1209_288_ROI'
    MAX_CENTER_DISTANCE = 6
    AREA_PCT = 0.1
    CENTER_PCT = 0.1
    MIN_PLOT_FRAMES = 0
    PLOT_FRAME_CONTOURS = True

    imgs_dir = set_and_check_folder(INPUT_FOLDER_NAME)
    output_img_dir = set_and_check_folder(IMAGE_OUTPUT_FOLDER_NAME, True)
    img_files, file_count = get_img_files_ordered(imgs_dir)
    frame_list = create_frame_list(img_files, file_count, imgs_dir,
        output_img_dir, IMAGE_FORMAT, PLOT_FRAME_CONTOURS)

    img_processing_time = time.time() - start_time # Log time it took to process images.
    
    # Create initial crystals
    crystal_tracking_list = []
    for obj in frame_list[0].crystalobjects:
        crystal_tracking_list.append(CrystalRecog(obj))
    print('Frame # 1:')
    print(f'Used count: {len(crystal_tracking_list)}')

    # Start from 1 here, because frame 0 / the first frame already done above
    for i in range(1,len(frame_list)):
        print(f'Frame # {i+1}:')
        c_central_list = frame_list[i].crystal_centers
        c_crystal_areas_list = frame_list[i].crystal_areas
        pre_frame_center_coord_count = len(c_central_list)
        for target_crys in crystal_tracking_list:
            # find the coordinates, index of said coordinates, and distance to last center point
            clostest_coord, index_closest, distance = closest(target_crys.center_arrays[len(target_crys.center_arrays) -1], c_central_list)
            if distance < MAX_CENTER_DISTANCE:
                # Find Crystal object corresponding to the central coord
                for crys in frame_list[i].crystalobjects:
                    if (crys.center_arr == clostest_coord).all():
                        if crys.area*(1-AREA_PCT) <= target_crys.areas[len(target_crys.areas) -1] <= crys.area*(1+AREA_PCT): 
                            target_crys.add_crystalobject(crys)
                            c_central_list.pop(index_closest)
                            c_crystal_areas_list.remove(crys.area)

            # else:
            #     print(f'Attempting to match {target_crys.cunt_num}')
            #     target_area = target_crys.areas[len(target_crys.areas) -1]
            #     for areas in itertools.combinations(c_crystal_areas_list, 2):
            #         if  target_area*(1-AREA_PCT) <= sum(areas) <= target_area*(1+AREA_PCT):
            #             print(f'found {sum(areas)} being equal to {target_area} ???')
            #             index_list = [c_crystal_areas_list.index(area) for area in areas]
            #             crys_list = []
            #             for crys in frame_list[i].crystalobjects:
            #                 for ind in index_list:
            #                     if crys.area == c_crystal_areas_list[ind]:
            #                         crys_list.append(crys)
            #             obj_center_contained = False
            #             print('**********')
            #             for crys in crys_list:
            #                 print(f'---------{len(crys_list)}')
            #                 if target_crys.min_y*(1-CENTER_PCT) <= crys.center_arr[1] <= target_crys.max_y*(1+CENTER_PCT) and target_crys.min_x*(1-CENTER_PCT) <= crys.center_arr[0] <= target_crys.max_x*(1+CENTER_PCT):
            #                     print(f'{target_crys.min_y*(1-CENTER_PCT)} <= {crys.center_arr[1]} <= {target_crys.max_y*(1+CENTER_PCT)}')
            #                     print(f'{target_crys.min_x*(1-CENTER_PCT)} <= {crys.center_arr[0]} <= {target_crys.max_x*(1+CENTER_PCT)}')
            #                     obj_center_contained = True
            #                 else:
            #                     obj_center_contained = False
            #                     break
            #             if obj_center_contained == True:
            #                 print('YAAASS QUEEN')
            #                 if len(crys_list) == 2:
            #                     print('doubleeee')
            #                     target_crys.add_dubble_crystalobject(crys_list[0], crys_list[1])
            #                 if len(crys_list) == 3:
            #                     print('T-T-T-T-tripple comboooooooo! Figure this shit out')
                        # From the crystals in crys_list, find max_x, max_y, etc.
                        # Check if these values are within min/max y and x of CrystalRecog
#

            #             # print(len(c_central_list))
        post_frame_center_coord_count = len(c_central_list)
        print('------------------------------------------------------')
        print(f'Frame # {i + 1}:')
        print(f'C coords went from {pre_frame_center_coord_count} to {post_frame_center_coord_count}  ')
        print(f'Used count: {pre_frame_center_coord_count - post_frame_center_coord_count }')
    crystal_linking_time = (time.time() - start_time) - img_processing_time # Log time it took to link crystals

    # crystal_tracking_count = len(crystal_tracking_list)
    # for i,b in enumerate(crystal_tracking_list):
    #     if b.cunt_num == 68:
    #         test_arr = np.array(b.s_contours[0])
    #         print(f'{min(test_arr[...,0])}')
    #         print(f'{max(test_arr[...,0])}')
    #         print(f'{min(test_arr[...,1])}')
    #         print(f'{max(test_arr[...,1])}')


    crystal_tracking_count = len(crystal_tracking_list)
    for i,b in enumerate(crystal_tracking_list):
        if b.cunt_num == 77:
            print(b.areas)
            print(b.center_arrays)
            print(np.array(b.s_contours[0]))
            # print(b.min_x, b.max_x, b.min_y, b.max_y)
        print(f'Plotting Crystal {i}/{crystal_tracking_count}', end = '\r')
        if b.c_count > MIN_PLOT_FRAMES:
            fig = plt.figure()
            fig.tight_layout()
            gs1 = fig.add_gridspec(nrows=3, ncols=2)
            fig_ax1 = fig.add_subplot(gs1[:-1, :])
            fig_ax1.title.set_text('Contours')
            for contour in b.s_contours:
                fig_ax1.plot(contour[...,0], contour[...,1])
            fig_ax1.invert_yaxis()
            fig_ax1.title.set_fontsize(12)
            fig.suptitle(t=f'#{b.cunt_num}; FU{b.c_count}/{file_count}', fontsize=12, va='top')

            fig_ax2 = fig.add_subplot(gs1[-1, :-1])
            fig_ax2.title.set_text('Area')
            fig_ax2.plot(b.areas)
            fig_ax2.title.set_fontsize(10)

            fig_ax3 = fig.add_subplot(gs1[-1, -1])
            fig_ax3.title.set_text('Mean Curvature')
            fig_ax3.plot(b.mean_curvatures)
            fig_ax3.title.set_fontsize(10)
            frames_used = ','.join(b.frames_used)
            fig.text(0.02, 0.02, 'FU: ' + frames_used, color='grey',fontsize=4)
            fig.subplots_adjust(left=None, bottom=None, right=None, top=0.90,
                wspace=0.3, hspace=0.5)
            fig.savefig(os.path.join(output_img_dir, f'newtest_img{b.cunt_num}.png'))
            plt.close()

    print('######################################################')
    print(f'img processing time: {img_processing_time} ')
    print(f'Crystal linking time : {crystal_linking_time}')
    print("Total runtime --- %s seconds ---" % (time.time() - start_time)) # To see how long program
    print('######################################################')


