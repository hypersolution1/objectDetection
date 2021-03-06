
process.env['TF_CPP_MIN_LOG_LEVEL'] = '3'

var cv = require('opencv4nodejs')

var fs = require('fs')
var readFileAsync = require('util').promisify(fs.readFile)

var ObjDetect = require('.')

;(async () => {

  var objdetect = await ObjDetect()

  var imgsrc = await cv.imreadAsync(`/var/store/test/truck1-1.jpg`)
  //var imgsrc = await cv.imreadAsync(`./in2.png`)

  var objs
  console.time('time')
  objs = await objdetect.detect(imgsrc)
  console.timeEnd('time')

  console.log(objs)

  for(let obj of objs) {
    var score = (obj.score*100).toFixed(0)
    imgsrc.drawRectangle(new cv.Point2(obj.x1, obj.y1), 
      new cv.Point2(obj.x2, obj.y2), new cv.Vec3(0, 255, 0), 4)
    imgsrc.putText(`${obj.label} ${score}%`,
      new cv.Point2(obj.x1, obj.y1 + 30), cv.FONT_HERSHEY_SIMPLEX, 1, new cv.Vec3(255, 255, 255), 5)      
    imgsrc.putText(`${obj.label} ${score}%`,
      new cv.Point2(obj.x1, obj.y1 + 30), cv.FONT_HERSHEY_SIMPLEX, 1, new cv.Vec3(0, 0, 0), 2) 
  }
  await cv.imwriteAsync("out.jpg", imgsrc)

})()
.catch(function (err) {
  console.log(err)
})
