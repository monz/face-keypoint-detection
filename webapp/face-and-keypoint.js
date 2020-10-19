function startCamera(callback) {
    const hdConstraints = {
        video: {width: {min: 1280}, height: {min: 720}},
        audio: false
    };

    // navigator.mediaDevices.getUserMedia({video: true, audio: false}) // use webcam without resolution constraint
    navigator.mediaDevices.getUserMedia(hdConstraints)
        .then(function(s) {
            stream = s;
            video.srcObject = s;
            video.play();
    })
        .catch(function(err) {
            console.log("An error occured! " + err);
    });

    video.addEventListener("canplay", onVideoCanPlay, false);

    if (callback) {
        callback()
    } else {
        onVideoStarted()
    }
}

function onVideoStarted() {
    streaming = true;
    startAndStop.innerText = 'Stop';
}

function stopCamera(callback) {
    if (!streaming) {return;}
  
    video.pause();
    video.srcObject = null;
    
    stream.getVideoTracks()[0].stop();

    if (callback) {
        callback()
    } else {
        onVideoStopped()
    }
}

function onVideoStopped() {
    streaming = false;    
    canvasOutputCtx.clearRect(0, 0, canvasOutput.width, canvasOutput.height);
    startAndStop.innerText = 'Start';
}

function onVideoCanPlay() {
    videoWidth = video.videoWidth;
    videoHeight = video.videoHeight;
    video.setAttribute("width", videoWidth);
    video.setAttribute("height", videoHeight);
    canvasOutput.width = videoWidth;
    canvasOutput.height = videoHeight;

    startVideoProcessing();
}

function startVideoProcessing() {
    if (!streaming) { console.warn("Please startup your webcam"); return; }

    let canvasInput = document.createElement('canvas');
    canvasInput.width = videoWidth;
    canvasInput.height = videoHeight;
    
    // load face detector model
    let faceClassifier = new cv.CascadeClassifier();
    faceClassifier.load('haarcascade_frontalface_default.xml');
  
    // prepare variables
    const FPS = 8;
    RED = new cv.Scalar(255, 0, 0, 255) // RGBA

    // for performance reasons, the image should be constructed with cv.CV_8UC4 
    // https://docs.opencv.org/3.4/dd/d00/tutorial_js_video_display.html

    let frame = new cv.Mat(videoHeight, videoWidth, cv.CV_8UC4);
    let capture = new cv.VideoCapture(video);

    // schedule the first one.
    setTimeout(processVideo, 0, FPS, frame, capture, faceClassifier);
}

function processVideo(fps, frame, capture, faceClassifier) {
    try {
        if (!streaming) {
            // clean and stop.
            frame.delete();
            return;
        }
        let begin = Date.now();

        // start processing.
        capture.read(frame);

        // detect faces
        const {faces, size} = detectFaces(frame, faceClassifier)

        // highlight face bounding box
        let frameToShow = markFacesInFrame(frame, faces, size, RED)

        // predict face keypoints on webcam live stream data\n",
        let faceKeypoints = detectFaceKeypoints(frame, faces, model, modelInputWidth, modelInputHeight, modelChannelCount);
        
        //  mark face key points in frame\n",
        markFaceKeypointsInFrame(frameToShow, faces, faceKeypoints, modelInputWidth, modelInputHeight, size);

        // show some frame
        cv.imshow('canvasOutput', frameToShow);

        // cleanup, prevents increasing memory usage and crash
        frameToShow.delete();

        // schedule the next recursive call to processVideo
        let delay = 1000/fps - (Date.now() - begin);
        setTimeout(processVideo, delay, fps, frame, capture, faceClassifier);
    } catch (err) {
        info.innerHTML = err;
    }
}

function markFacesInFrame(frame, faces, detectionSize, color) {
    let outFrame = frame.clone()
    let xRatio = videoWidth/detectionSize.width;
    let yRatio = videoHeight/detectionSize.height;

    for (let i = 0; i < faces.length; ++i) {
        let face = faces[i];
        let point1 = new cv.Point(face.x*xRatio, face.y*yRatio);
        let point2 = new cv.Point(face.x*xRatio + face.width*xRatio, face.y*yRatio + face.height*yRatio);
        cv.rectangle(outFrame, point1, point2, color, 2, cv.LINE_AA, 0);
    }

    return outFrame
}

function detectFaces(frame, faceClassifier) {
    // convert frame to gray-scale
    let frameGray = new cv.Mat(videoHeight, videoWidth, cv.CV_8UC1);
    cv.cvtColor(frame, frameGray, cv.COLOR_RGBA2GRAY);
    
    // downsample (to half image size) input image?!
    // downsample twice when video width is greater than 320px
    // when not downsampling, face detection is too slow
    let faceMat = new cv.Mat();
    cv.pyrDown(frameGray, faceMat);
    if (videoWidth > 320) {
        cv.pyrDown(faceMat, faceMat);
    }
    let size = faceMat.size();
    
    // for debug, when not using pyrDown!
    // faceMat = frameGray.clone()
    // size = faceMat.size();
    
    // detect faces
    let faceVect = new cv.RectVector();
    let faces = [];
    faceClassifier.detectMultiScale(faceMat, faceVect);
    for (let i = 0; i < faceVect.size(); i++) {
        let face = faceVect.get(i);
        faces.push(new cv.Rect(face.x, face.y, face.width, face.height));
    }

    faceMat.delete();
    faceVect.delete();
    frameGray.delete();
    
    // return multiple values in javascript
    // https://stackoverflow.com/questions/2917175/return-multiple-values-in-javascript
    return {
        faces: faces,
        size: size,
    };
}

function detectFaceKeypoints(frame, faces, model, modelInputWidth, modelInputHeight, modelChannelCount) {
    // convert frame to gray-scale
    let frameGray = new cv.Mat(videoHeight, videoWidth, cv.CV_8UC1);
    cv.cvtColor(frame, frameGray, cv.COLOR_RGBA2GRAY);
    
    let batchSize = 1;
    let predictions = [];
    for (let i = 0; i < faces.length; ++i) {
        let faceRect = faces[i];

        // get face patch; region of interest
        faceRoi = frameGray.roi(faceRect); // type(face) = cv.Rect
        
        // resize ROI to model input size
        face = new cv.Mat()
        cv.resize(faceRoi, face, new cv.Size(modelInputWidth, modelInputHeight), 0, 0, cv.INTER_AREA); 
        
        // normalize input, add channel dimension
        normalizedFace = tf.tensor(face.data, [batchSize, modelInputWidth, modelInputHeight, modelChannelCount]).div(255.0);

        // predict face key points
        predictions.push(model.predict(normalizedFace));

        faceRoi.delete();
        face.delete();
    }
    frameGray.delete();

    return predictions;
}

function markFaceKeypointsInFrame(frame, faces, faceKeyPoints, modelInputWidth, modelInputHeight, detectionSize) {
    let radius = 2;
    let borderThickness = -1;  // -1 will fill the circle with given color

    let xRatio = videoWidth/detectionSize.width;
    let yRatio = videoHeight/detectionSize.height;
    
    for (let i = 0; i < faces.length; ++i) {
        let face = faces[i];
        let xKeypointRatio = face.width/modelInputWidth;
        let yKeypointRatio = face.height/modelInputHeight;

        let faceKeyPointCount = faceKeyPoints[i].shape[1]
        for (let j = 0; j < faceKeyPointCount; ++j) {
            let xPoint = faceKeyPoints[i].dataSync()[j];
            let yPoint = faceKeyPoints[i].dataSync()[++j];
            let center = new cv.Point(
                Math.round(face.x*xRatio + (xPoint*xKeypointRatio)*xRatio), // x
                Math.round(face.y*yRatio + (yPoint*yKeypointRatio)*yRatio)) // y
            cv.circle(frame, center, radius, RED, borderThickness);
        }
    }
    return frame;
}

function checkFeatures(info, features) {
    var wasmSupported = true, webrtcSupported = true;
    if (features.webrtc) {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        webrtcSupported = false;
      }
    }
    if (features.wasm && !window.WebAssembly) {
      wasmSupported = false;
    }
  
    if (!webrtcSupported || !wasmSupported) {
      var text = "Your web browser doesn't support ";
      var len = text.length;
      if (!webrtcSupported) {
        text += "WebRTC";
      }
      if (!wasmSupported) {
        if (text.length > len) {
          text += " and ";
        }
        text += "WebAssembly"
      }
      text += ".";
      info.innerHTML = text;
      return false;
    }
  
    return true;
  }

async function loadModel (modelUrl) {
    console.log('Loading tensorflow-model');
    info.innerHTML = 'Loading Neural Network Model...';

    const model = await tf.loadLayersModel(modelUrl);
    
    console.log('tensorflow-model ready');
    info.innerHTML = '';

    return model
}

const MODEL_URL = './face_keypoint_reduced_model/model.json';

let videoWidth;
let videoHeight;
let streaming = false;
let stream = null;
let video = document.getElementById('video');
let canvasOutput = document.getElementById('canvasOutput');
let canvasOutputCtx = canvasOutput.getContext('2d');
let info = document.getElementById('info');
let modelInputWidth = 96; // CNN input width
let modelInputHeight = 96; // CNN input height
let modelChannelCount = 1; // CNN input channel count
let model = -1; // placeholder for tensorflow model

async function opencvIsReady() {
    console.log('OpenCV.js is ready');

    model = await loadModel(MODEL_URL)

    // check if browser supports necessary technology
    var featuresSupport = checkFeatures(info, {webrtc: true, wasm: true});
    if (!featuresSupport) {
        console.log('Required browser features are not supported.');
        return;
    }
    info.innerHTML = '';

    // start webcam-stream
    startCamera();
}

// info
// https://www.html5rocks.com/en/tutorials/getusermedia/intro/
// https://www.tensorflow.org/js/guide/tensors_operations
// https://www.geeksforgeeks.org/how-to-convert-a-float-number-to-the-whole-number-in-javascript/
// https://js.tensorflow.org/api/latest/
// https://developer.mozilla.org/de/docs/Web/JavaScript/Reference/Statements/async_function
// https://docs.opencv.org/3.4/de/d06/tutorial_js_basic_ops.html
// https://github.com/tensorflow/tfjs/tree/master/tfjs-converter
// https://github.com/tensorflow/tfjs
// https://docs.opencv.org/master/d4/da1/tutorial_js_setup.html
// https://huningxin.github.io/opencv.js/samples/index.html
// https://docs.opencv.org/3.4/dd/d00/tutorial_js_video_display.html
// https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API/Taking_still_photos