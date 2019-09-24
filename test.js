
process.env['TF_CPP_MIN_LOG_LEVEL'] = '3'

var fs = require('fs')
var readFileAsync = require('util').promisify(fs.readFile)

var cv = require('opencv4nodejs')
var sharp = require('sharp')

var TFModel = require('libtf')

var model = TFModel({
  'gpu_memory_fraction': 0.25,
})

async function pad_image(img, padheight, padwidth) {
  var height = img.rows
  var width = img.cols
  var channel = img.channels

  if(height > padheight || width > padwidth) {
    width = (width > padwidth) ? padwidth : width
    height = (height > padheight) ? padheight : height
    img = await img.getRegion(new cv.Rect(0, 0, width, height)).copyAsync()
  }
  if(height < padheight || width < padwidth) {
    var img = await img.copyMakeBorderAsync(0, padheight - height, 0, padwidth - width, cv.BORDER_CONSTANT, new cv.Vec3(0, 0, 0))
  }
 
  return img
}

async function imgResize(img, width, height) {
  var sharpresize = await sharp(img.getData(), {
    raw: {
        width: img.cols,
        height: img.rows,
        channels: 3
    }
  })
  .resize({ width, height })
  .toBuffer({ resolveWithObject: true })
  return new cv.Mat(sharpresize.data, sharpresize.info.height, sharpresize.info.width, cv.CV_8UC3)
}

;(async () => {

  await model.load('./ssd_inception_v2_coco_2018_01_28.pb') //faster_rcnn_resnet101_coco_2018_01_28

  var labelmap = JSON.parse(await readFileAsync("./labelmap.json"))
  var labels = {}
  for(var l of labelmap) {
    labels[l.id] = l.display_name
  }

  var imgsrc = await cv.imreadAsync(`/var/store/test/truck1-1.jpg`)
  var ratio = imgsrc.cols / imgsrc.rows
  var imginput = imgsrc //await pad_image(await imgResize(imgsrc, 600), 600, 600) // await imgResize(imgsrc, 300, 300) 
  //await cv.imwriteAsync("imginput.jpg", imginput)
  var input = {
    "image_tensor": {
      "dim": [1, imginput.rows, imginput.cols, 3],
      "data": imginput.getData(),
    },
    
  }
  var val
  console.time('time')
  val = await model.execute(input, ["detection_boxes", "detection_scores", "num_detections", "detection_classes"])
  console.timeEnd('time')
  console.time('time')
  val = await model.execute(input, ["detection_boxes", "detection_scores", "num_detections", "detection_classes"])
  console.timeEnd('time')
  console.time('time')
  val = await model.execute(input, ["detection_boxes", "detection_scores", "num_detections", "detection_classes"])
  console.timeEnd('time')
  console.time('time')
  val = await model.execute(input, ["detection_boxes", "detection_scores", "num_detections", "detection_classes"])
  console.timeEnd('time')

  var num_detections = val.num_detections.data[0]
  for (var i = 0; i < num_detections; i++) {
    var detection_scores = val.detection_scores.data[i]
    if(detection_scores > 0.3) {
      var score = (detection_scores * 100).toFixed(0)
      var detection_boxes = val.detection_boxes.data.subarray(4*i,4*(i+1))
      var [ymin, xmin, ymax, xmax] = detection_boxes
      var label = labels[val.detection_classes.data[i]]
      console.log(label, score)
      imgsrc.drawRectangle(new cv.Point2(xmin*imgsrc.cols, ymin*imgsrc.rows), 
        new cv.Point2(xmax*imgsrc.cols, ymax*imgsrc.rows), new cv.Vec3(0, 255, 0), 4)
      imgsrc.putText(`${label}`,
        new cv.Point2(xmin*imgsrc.cols, ymin*imgsrc.rows + 30), cv.FONT_HERSHEY_SIMPLEX, 1, new cv.Vec3(0, 0, 0), 4)


    }
  }
  await cv.imwriteAsync("out.jpg", imgsrc)
  //console.log(val)
})()
.catch(function (err) {
  console.log(err)
})
