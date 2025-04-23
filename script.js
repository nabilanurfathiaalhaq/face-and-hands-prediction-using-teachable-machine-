let model;
let webcam;

async function init() {
    model = await tf.loadLayersModel('model.json');
    webcam = await tf.data.webcam(document.getElementById('webcam'));
    predict();
}

async function predict() {
    while(true){
        const img = await webcam.capture();
        const processedImg = tf.tidy(() => {
            return img.resizeNearestNeighbor([224,224])
            .toFloat()
            .div(tf.scalar(255))
            .expandDims()
        });
        const predictions = await model.predict(processedImg);
        const predictedClass = predictions.argMax(-1).dataSync()[0];
        let label;
        if(predictedClass===0){
            label ="Hands";
        } else if(predictedClass===1){
            label = "Face";
        } else {
            label = "Unknown";
        }
        document.getElementById('prediction').innerText = `Predicted Class: ${label}`;
        img.dispose();
        processedImg.dispose();
        predictions.dispose();
        await tf.nextFrame();
    }
}
init();