---
layout: post
title: 目标检测--Faster RCNN2
tag: object detection
category: object_detection
comments: true
blog: true
data: 2016-09-29
---
### 写在前面的话　　

今天这篇博客主要讲解如何利用faster-rcnn训练自己的数据，当然这不是原创，也是自己踩坑之后得到的经验，不要方，慢慢来！　　


## 训练自己的数据集　　　　

首先我们来分析一下pascal_voc数据集的目录结构，众所周知，该数据集可以做目标检测，目标分割，识别等任务，因此对于目标检测而言，我们仅仅需要其中几个文件夹，带我娓娓道来。下面是VOC2007的数据集文件
目录结构，我们只需要文件夹*Annotations*, *ImageSets*, *JPEGImages*, 其中*JPEGImages*中主要存放原始图片，无需多说;关于*Annotations*和*ImageSets*我们需要在下面章节中详细讲解。

└── VOC2007　　

    ├── Annotations　　

    ├── ImageSets　　

    │   ├── Layout　　

    │   ├── Main　　

    │   └── Segmentation　　

    ├── JPEGImages　　

    ├── SegmentationClass　　

    └── SegmentationObject　　　　

### Annotations　& ImageSets

*Annotations*中主要存放xml文件，每个xml文件对应一张图片，而每个xml中存放为一张图片中各个目标的位置，类别等信息，如下述代码所述:  

> xml格式

```xml
<annotation>
	<folder>VOC2007</folder> # 必须有
	<filename>000005.jpg</filename>　#　必须有
	<source>　# 可有可无
		<database>The VOC2007 Database</database>
		<annotation>PASCAL VOC2007</annotation>
		<image>flickr</image>
		<flickrid>325991873</flickrid>
	</source>
	<owner>　# 可有可无
		<flickrid>archintent louisville</flickrid>
		<name>?</name>
	</owner>
	<size>　# 表示图像大小
		<width>500</width>
		<height>375</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>　# 用于分割
	<object>　# 目标信息，类别，bbox信息
		<name>chair</name>
		<pose>Rear</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>263</xmin>
			<ymin>211</ymin>
			<xmax>324</xmax>
			<ymax>339</ymax>
		</bndbox>
	</object>
	<object>
		<name>chair</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>165</xmin>
			<ymin>264</ymin>
			<xmax>253</xmax>
			<ymax>372</ymax>
		</bndbox>
	</object>
	<object>
		<name>chair</name>
		<pose>Unspecified</pose>
		<truncated>1</truncated>
		<difficult>1</difficult>
		<bndbox>
			<xmin>5</xmin>
			<ymin>244</ymin>
			<xmax>67</xmax>
			<ymax>374</ymax>
		</bndbox>
	</object>
	<object>
		<name>chair</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>241</xmin>
			<ymin>194</ymin>
			<xmax>295</xmax>
			<ymax>299</ymax>
		</bndbox>
	</object>
	<object>
		<name>chair</name>
		<pose>Unspecified</pose>
		<truncated>1</truncated>
		<difficult>1</difficult>
		<bndbox>
			<xmin>277</xmin>
			<ymin>186</ymin>
			<xmax>312</xmax>
			<ymax>220</ymax>
		</bndbox>
	</object>
</annotation>
```

代码中注释我们需要生成的信息，即我们需要生成类似表示图片中目标信息的文件。对于具有标注信息的数据集，我们修改其格式即可，而如果是自己的数据，则需要自己标注框，此处推荐[lableImag](https://github.com/saicoco/object_labelImg),
它可以通过我们自己标注物体，自行生成对应xml文件，而关于它生成的文件有点小毛病，即`filename`对应的图片名没包含后缀jpg,因此需要写代码来修改它，当然使用linux命令也可以实现，此处为我的代码:　　

> 修改xml文件

```python
import xml.dom.minidom as xdm
import glob

path = './Annotations/'

items = glob.glob(path+'*.xml')

train_txt = 'train.txt'

for item in items:
    dom = xdm.parse(item)
    root = dom.documentElement

    # change folder name
    folder_dom = root.getElementsByTagName('folder')
    print folder_dom[0].firstChild.data
    folder_dom[0].firstChild.data = 'VOC2007'

    # change image name

    image_dom = root.getElementsByTagName('filename')
    print image_dom[0].firstChild.data

    with open(train_txt, 'a+') as f:
        f.write(image_dom[0].firstChild.data)
        f.write('\n')

    image_dom[0].firstChild.data = image_dom[0].firstChild.data + '.jpg'
    with open(item, 'w') as f:
        dom.writexml(f, encoding='utf-8')

```  

上述代码在修改xml的同时，生成对应图片的列表，而该列表便是存放于*ImageSets/Main*中的文件，其中存放内容为图片名称但是不包含后缀jpg,命名规则为train.txt, trainval.txt
val.txt等，可以参看VOC2007下的数据列表格式。将以上得到的文件放于VOC2007下对应文件内，即替换原有文件，然后就做成了voc格式的目标检测数据集。　　

### 代码部分　　

接下来就是修改代码中对应位置的值就可以。　　

首先是'py-faster-rcnn/lib/datasets/pascal_voc.py`，需要修改如下部分:  

```python
class pascal_voc(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'voc_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        self._classes = ('__background__', 'dog', 'person', 'cat', 'car')
        # self._classes = ('__background__', # always index 0
        #                  'aeroplane', 'bicycle', 'bird', 'boat',
        #                  'bottle', 'bus', 'car', 'cat', 'chair',
        #                  'cow', 'diningtable', 'dog', 'horse',
        #                  'motorbike', 'person', 'pottedplant',
        #                  'sheep', 'sofa', 'train', 'tvmonitor')
```  

在`self._classes`中添加你需要检测的类别，这里我有四个类别再加上背景总共五个类别，然后便是修改对应训练模型的网络文件，这里我使用的
是ZF模型，因此需要修改路径`py-faster-rcnn/models/pascal_voc/ZF/`中的对应训练方式的`*.prototxt`，这里我选用end2end,贴上修改的部分：　　

```
name: "ZF"
layer {
  name: 'input-data'
  type: 'Python'
  top: 'data'
  top: 'im_info'
  top: 'gt_boxes'
  python_param {
    module: 'roi_data_layer.layer'
    layer: 'RoIDataLayer'
    param_str: "'num_classes': 5" # 按训练集类别改(类别+背景)
  }
}

###############################################################  

layer {
  name: 'roi-data'
  type: 'Python'
  bottom: 'rpn_rois'
  bottom: 'gt_boxes'
  top: 'rois'
  top: 'labels'
  top: 'bbox_targets'
  top: 'bbox_inside_weights'
  top: 'bbox_outside_weights'
  python_param {
    module: 'rpn.proposal_target_layer'
    layer: 'ProposalTargetLayer'
    param_str: "'num_classes': 5" # 此处修改为类别+1
  }
}

###############################################################

layer {
  name: "cls_score"
  type: "InnerProduct"
  bottom: "fc7"
  top: "cls_score"
  param { lr_mult: 1.0 }
  param { lr_mult: 2.0 }
  inner_product_param {
    num_output: 5 # 按照类别改，类别＋1,即包括背景+类别
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}

################################################################
layer {
  name: "bbox_pred"
  type: "InnerProduct"
  bottom: "fc7"
  top: "bbox_pred"
  param { lr_mult: 1.0 }
  param { lr_mult: 2.0 }
  inner_product_param {
    num_output: 20 # 按类别数改，(类别数+1)*4
    weight_filler {
      type: "gaussian"
      std: 0.001
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
```  

至此代码修改完毕，具体参数配置可以修改`train_net.py`，`faster_rcnn_end2end.sh`中内容。对于xml中标注信息的获取，除了手工标注，利用
opencv等进行位置获取也是可以的，总而言之，文件夹格式正确，便可以训练我们自己的数据。(ps:我貌似找着了如何使得连续代码块不报错的方法了，在每个代码块的前方加入'>'就可以了，哈哈)

### 写在后面的话　　

今天跟一北航的聊天，聊到华为硕士年薪28.8万，惊了个呆，期间还吐槽了学术圈，瞎扯了一下接下来的一年。接下来的一年肯定得做自己的事情了，
我是清都山水郎，天教懒慢带疏狂。曾批给露支风券，累奏流云借月章。诗万首，酒千觞，几曾着眼看侯王。玉楼金阙慵归去，且插梅花醉洛阳……
