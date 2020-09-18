# This is a visualization tool of vehicle re-identification.
# Step 1: Save the matched test image IDs for each query image as a text file, where each line contains a test image ID ranked in terms of distance score in ascending order. Name each text file as '%06d.txt' % <query_image_ID>. We assume that the top-50 matched test images are displayed. An example is given in "./dist_example/".
# Step 2: Run "python visualize.py".
# Step 3: Input the path of the directory containing all text files at "Txt Dir:" (end with '/'). An example is given as "./dist_example/".
# Step 4: Click "Load".
# The query image is shown on the top left. The corresponding test images are shown on the right.
# For each image, the image ID is shown on the top left corner. 
# Click "<< Prev" to return to the previous query.
# Click "Next >>" to advance to the next query.
# Enter the query no. and click "Go" to jump to the corresponding query.  

from __future__ import division
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import glob
import cv2
import threading
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

# class get_distance(threading.Thread):
#     def __init__(self, *args, **kwargs):
#         threading.Thread.__init__(self, *args, **kwargs)
#         self.daemon = True
#         self.start()
#         self.img =[]

#     def getfile(self):
#         return self.img

#     def run(self):
        #Run Model
        # cmd = 'python D:/environments/AI_city_reID_py36/AICity-track2-Re-id-csmvl310/main.py --mode aicity --data_path "D:/environments/AI_city_reID_py36/AICity-track2-Re-id-csmvl310_v2_org/AIC20_ReID" --query_image ' + root.filename[-10:]
        # os.system(cmd)
        
class VisTool():
    def __init__(self, master):
        # set up the main frame
        self.parent = master
        self.parent.title("Re-ID result")
        self.frame = Frame(self.parent)
        self.frame.pack(fill=BOTH, expand=1)
        self.parent.resizable(width=FALSE, height=FALSE)

        # initialize global state
        self.txtDir = ''
        self.txtList = []
        self.prbList = []
        self.outDir = ''
        self.cur = 0
        self.total = 0
        self.category = ''
        self.imagename = ''
        self.labelfilename = ''
        self.filename = ''
        self.tkimg = None
        # self.tkimg1 = [None] * 10000
        self.img = []

        # ----------------- GUI stuff ---------------------
        # dir entry & load
        self.label = Label(self.frame, text="Txt Dir:")
        self.label.grid(row=0, column=0, sticky=E)
        self.entry = Entry(self.frame)
        self.entry.grid(row=0, column=1, sticky=W + E)
        self.ldBtn = Button(self.frame, text="載入結果", command=self.loadDir)
        self.ldBtn.grid(row=0, column=2, sticky=W + E)

        # Choose query image & show gallery distance
        self.ldBtn = Button(self.frame, text="選擇檔案", command=lambda: [self.showQuery(), threading.Thread(target=tool.load_Dist_result).start()])
        self.ldBtn.grid(row=1, column=2, sticky=W + E)

        # Rank the result
        self.ldBtn = Button(self.frame, text="進行排序", command=lambda: threading.Thread(target=tool.rank_dist).start())
        self.ldBtn.grid(row=2, column=2, sticky=W + E)

        # main panel
        self.mainPanel = Canvas(self.frame, cursor='arrow')
        self.parent.bind("a", self.prevPrb)  # press 'a' to go backforward
        self.parent.bind("d", self.nextPrb)  # press 'd' to go forward
        self.mainPanel.grid(row=1, column=1, rowspan=4, sticky=W + N)

        # control panel for image navigation
        self.ctrPanel = Frame(self.frame)
        self.ctrPanel.grid(row=5, column=1, columnspan=2, sticky=W + E)
        self.prevBtn = Button(self.ctrPanel, text='<< Prev', width=10, command=self.prevPrb)
        self.prevBtn.pack(side=LEFT, padx=5, pady=3)
        self.nextBtn = Button(self.ctrPanel, text='Next >>', width=10, command=self.nextPrb)
        self.nextBtn.pack(side=LEFT, padx=5, pady=3)
        self.progLabel = Label(self.ctrPanel, text="Progress:     /    ")
        self.progLabel.pack(side=LEFT, padx=5)
        self.tmpLabel = Label(self.ctrPanel, text="Go to Query No.")
        self.tmpLabel.pack(side=LEFT, padx=5)
        self.idxEntry = Entry(self.ctrPanel, width=5)
        self.idxEntry.pack(side=LEFT)
        self.goBtn = Button(self.ctrPanel, text='Go', command=self.gotoPrb)
        self.goBtn.pack(side=LEFT)

        # panel for query image
        self.prbPanel = Frame(self.frame, border=10)
        self.prbPanel.grid(row=1, column=0, rowspan=5, sticky=N)
        self.tmpLabel2 = Label(self.prbPanel, text="Query image:")
        self.tmpLabel2.pack(side=TOP, pady=5)
        self.prbLabels = []
        self.prbLabels.append(Label(self.prbPanel))
        self.prbLabels[-1].pack(side=TOP)

        #scrollbar control
        self.v_scroll = Scrollbar(self.mainPanel, orient='vertical')

        #panel for gallery images
        self.gallery_text1 = Text(self.mainPanel, width=1200, height=750, yscrollcommand= self.v_scroll.set)
        self.gallery_text2 = Text(self.mainPanel, width=1200, height=750, yscrollcommand= self.v_scroll.set)
        self.gallery_text = Text(self.mainPanel, width=1200, height=750, yscrollcommand= self.v_scroll.set)

        # display mouse position
        self.disp = Label(self.ctrPanel, text='')
        self.disp.pack(side=RIGHT)

        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(4, weight=1)
    
    def create_text(self):
        self.gallery_text.delete(1.0, END)
        self.gallery_text = Text(self.mainPanel, width=1200, height=750, yscrollcommand= self.v_scroll.set)
        # self.gallery_text.pack()

    def loadDir(self):
        # get txt file list
        self.txtDir = os.path.join(self.entry.get())
        self.txtList = glob.glob(os.path.join(self.txtDir, '*.txt'))
        self.txtList.sort(key=lambda x: int(x[-10:-4]))

        if len(self.txtList) == 0:
            print ('No text file found in the specified directory!')
            return

        # default: the 1st image in the collection
        self.cur = 1
        self.total = len(self.txtList)
        self.loadImage()
        print ('%d images loaded' % self.total)
    
    def showQuery(self):
        root.filename =  filedialog.askopenfilename(initialdir = "D:\\environments\\AI_city_reID_py36\\AIC20_track2\\AIC20_ReID\\tool\\image_query", 
        title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
        self.filename = root.filename

        #load query image & show on panel
        self.List = []
        im = cv2.imread(self.filename)
        # save_path = 'D:\\environments\\AI_city_reID_py36\\AICity-track2-Re-id-csmvl310\\AIC20_ReID\\visualize_query_image\\' + self.filename[-10:]
        # cv2.imwrite(save_path, im)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (150, 150))
        cv2.putText(im, self.filename[-10:-4], (10,20), cv2.FONT_HERSHEY_COMPLEX,0.5,(255,128,0),1)
        im = Image.fromarray(im)
        self.List.append(ImageTk.PhotoImage(im))
        self.prbLabels[0].config(image=self.List[-1], width=150, height=150)

    def load_Dist_result(self):
        self.gallery_text1.pack()
        # del self.tkimg1
        self.gallery_text1.delete(1.0, END)
        self.tkimg1 = []
        
        self.img =[]
        dist = np.load('final_dist.npy')

        #display gallery image & distance
        result_list = []
        for i in range(18290):
            result_list.append("../image_test/%06d.jpg" % int(i+1))
        visRange = len(result_list)
        visRange = min(visRange, 2000)
        myrange = np.array(range(visRange))

        myimage = []
        self.img = Image.new('RGB', (1500, 750))
        for i in range(visRange):
            im = cv2.imread(result_list[i])
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (150, 150))
            cv2.putText(im, result_list[i][-10:-4], (10,20), cv2.FONT_HERSHEY_COMPLEX,0.5,(255,128,0),1)
            cv2.putText(im, dist[int(self.filename[-10:-4])-1][i].astype('str'), (35,145), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),1)
            myimage.append(Image.fromarray(im))

            self.v_scroll.pack(side=RIGHT, fill=Y)
            # lb = Listbox(self.mainPanel, yscrollcommand= self.v_scroll.set)

            self.tkimg1.append(ImageTk.PhotoImage(myimage[i]))
            self.gallery_text1.image_create(END, image=self.tkimg1[i])
            self.v_scroll.config(command=self.gallery_text1.yview)

            # self.mainPanel.create_image(0, 0, image=self.tkimg, anchor=NW)
            # lb.insert(END, self.tkimg)
            # lb.pack(side=RIGHT)
            # self.v_scroll.config(command=lb.yview)

        # self.tkimg = ImageTk.PhotoImage(self.img)
        # self.mainPanel.config(width=self.tkimg.width(), height=self.tkimg.height())
        # self.mainPanel.create_image(0, 0, image=self.tkimg, anchor=NW)

        # for i in tqdm(range(dist.shape[0])):
        #     txt_list = np.zeros((100))
        #     for j in range(txt_list.shape[0]):
        #         txt_list[j] = int(np.argmin(dist[i])) + 1
        #         dist[i][int(txt_list[j]) - 1] = 100000
        #     txt_list = pd.DataFrame(txt_list, dtype=np.int32)
        #     txt_list.to_csv('save/{:0>6d}.txt'.format(i + 1), header=0, index=0)


        #######################################################Run model###########################################################
        # Threader(name='RunModel')

        #display result
        # result_path = "D:/environments/AI_city_reID_py36/AIC20_track2/AIC20_ReID/tool/save/" + root.filename[-10:-4] + ".txt"
        # result_list = []
        # with open(result_path) as f:
        #     for line in f.readlines():
        #         result_list.append("../image_test/%06d.jpg" % int(line))
        # visRange = len(result_list)
        # visRange = min(visRange, 50)
        # myrange = np.array(range(visRange))
        # myimage = []

        # for i in myrange:
        #     im = cv2.imread(result_list[i])
        #     im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        #     im = cv2.resize(im, (150, 150))
        #     cv2.putText(im, result_list[i][-10:-4], (10,20), cv2.FONT_HERSHEY_COMPLEX,0.6,(255,128,0),1)
        #     myimage.append(Image.fromarray(im))

        # self.img = Image.new('RGB', (1500, 750))

        # ss = 10
        # for i in range(visRange):
        #     row = int(i / ss)
        #     tlx = (i - ss * row) * 150
        #     tly = 150 * row
        #     shape = myimage[i].size
        #     self.img.paste(myimage[i], (tlx, tly, tlx + shape[0], tly + shape[1]))
        # self.tkimg = ImageTk.PhotoImage(self.img)
        # self.mainPanel.config(width=self.tkimg.width(), height=self.tkimg.height())
        # self.mainPanel.create_image(0, 0, image=self.tkimg, anchor=NW)
        # self.progLabel.config(text="%04d/%04d" % (self.cur, self.total))
        ##########################################################################################################################
    
    def rank_dist(self):  
        # del self.tkimg1
        self.gallery_text1.delete(1.0, END)
        self.tkimg1 = []

        dist = np.load('final_dist.npy')
        
        ranked_list = []
        distance_list = []
        for j in range(2000):
            minGallery = int(np.argmin(dist[int(self.filename[-10:-4])-1])) + 1
            distance_list.append(dist[int(self.filename[-10:-4])-1][minGallery - 1])
            dist[int(self.filename[-10:-4])-1][minGallery - 1] = 1000
            ranked_list.append("../image_test/%06d.jpg" % minGallery)
        visRange = len(ranked_list)

        myimage = []
        for i in range(visRange):
            im = cv2.imread(ranked_list[i])
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (150, 150))
            cv2.putText(im, ranked_list[i][-10:-4], (10,20), cv2.FONT_HERSHEY_COMPLEX,0.5,(255,128,0),1)
            cv2.putText(im, distance_list[i].astype('str'), (35,145), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),1)
            myimage.append(Image.fromarray(im))

            self.v_scroll.pack(side=RIGHT, fill=Y)

            self.tkimg1.append(ImageTk.PhotoImage(myimage[i]))
            self.gallery_text1.image_create(END, image=self.tkimg1[i])
            self.v_scroll.config(command=self.gallery_text1.yview)
        print("display finish !!!!!")
        

    def loadImage(self):
        txtPath = self.txtList[self.cur - 1]

        mylist = []
        mylist.append("../image_query/" + txtPath[-10:-4] + ".jpg")
        with open(txtPath) as f:
            for line in f.readlines():
                mylist.append("../image_test/%06d.jpg" % int(line))

        self.tmp = []
        self.prbList = []

        # load query image
        f = mylist[0]
        im = cv2.imread(f)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (150, 150))
        cv2.putText(im, f[-10:-4], (10,20), cv2.FONT_HERSHEY_COMPLEX,0.5,(255,128,0),1)
        #cv2.rectangle(im, (0, 0), (149, 149), (0,0,255), 8)
        im = Image.fromarray(im)
        self.tmp.append(im)
        self.prbList.append(ImageTk.PhotoImage(self.tmp[-1]))
        # print("image : ", self.prbList[-1])
        self.prbLabels[0].config(image=self.prbList[-1], width=150, height=150)

        # load gallery images
        mylist = mylist[1:]
        # print("mylist : ", mylist)
        siz = len(mylist)
        siz = min(siz, 50)
        myrange = np.array(range(siz))
        myimage = []

        for i in myrange:
            im = cv2.imread(mylist[i])
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (150, 150))
            cv2.putText(im, mylist[i][-10:-4], (10,20), cv2.FONT_HERSHEY_COMPLEX,0.6,(255,128,0),1)

            #if int(mylist[i][-10:-4]) in self.ground_truths[int(f[-10:-4]) - 1]:
            #    cv2.rectangle(im, (0, 0), (149, 149), (0,255,0), 8)
            #else:
            #    cv2.rectangle(im, (0, 0), (149, 149), (255,0,0), 8)

            myimage.append(Image.fromarray(im))

        self.img = Image.new('RGB', (1500, 750))

        ss = 10
        for i in range(siz):
            row = int(i / ss)
            tlx = (i - ss * row) * 150
            tly = 150 * row
            shape = myimage[i].size
            self.img.paste(myimage[i], (tlx, tly, tlx + shape[0], tly + shape[1]))
            
            self.tkimg = ImageTk.PhotoImage(self.img)
            self.mainPanel.config(width=self.tkimg.width(), height=self.tkimg.height())
            self.mainPanel.create_image(0, 0, image=self.tkimg, anchor=NW)
            self.progLabel.config(text="%04d/%04d" % (self.cur, self.total))

    def prevPrb(self, event=None):
        if self.cur > 1:
            self.cur -= 1
            self.loadImage()

    def nextPrb(self, event=None):
        if self.cur < self.total:
            self.cur += 1
            self.loadImage()

    def gotoPrb(self):
        idx = int(self.idxEntry.get())
        if 1 <= idx and idx <= self.total:
            self.cur = idx
            self.loadImage()


if __name__ == '__main__':
    root = Tk()
    tool = VisTool(root)
    root.resizable(width=True, height=True)
    root.mainloop()
