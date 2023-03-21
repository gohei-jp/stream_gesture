
const video = document.getElementById('video');

function startCamera() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
        })
        .catch(error => {
            console.error('Error accessing camera: ', error);
        });
    }
}
startCamera();
const ws = new WebSocket('ws://localhost:8000');
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');

ws.addEventListener('open', () => {
    console.log('WebSocket connection opened.');

    setInterval(() => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);
        const imageData = canvas.toDataURL('image/jpeg', 0.8);
        ws.send(imageData);
    }, 200);
});
const processedImage = document.getElementById('processed-image');

ws.addEventListener('message', (event) => {
    processedImage.src = event.data;
});
