import React, { useEffect, useRef, useState } from "react";
import axios from 'axios'
const Recognition = () => {
  const canvasRef = useRef();
  const imageRef = useRef();
  const videoRef = useRef();

  const [result, setResult] = useState("");

  useEffect(() => {
    async function getCameraStream() {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: true,
      });
  
      if (videoRef.current) {      
        videoRef.current.srcObject = stream;
      }
    };
  
    getCameraStream();
  }, []);
  async function fetchMoviesJSON() {
    let headers = new Headers();
    // headers.append('Content-Type', 'application/json');
    // headers.append('Accept', 'application/json');
    // headers.append('Origin','http://localhost:8000');

    const response = await axios.get(
      'http://localhost:8000/test',
    );
    const movies = await response.json();
    return movies;
  }
  var FPS = 30
  useEffect(() => {
    const interval = setInterval(async () => {
      captureImageFromCamera();
      // fetchMoviesJSON().then(movies => {
      //   console.log(movies); // fetched movies
      // });
      // fetch("http://localhost:8000/test").then((Response) => {
      //       return Response.json()
      //   }).then((data) => {
      //       console.log(data);
      //   })

      if (imageRef.current) {
        const formData = new FormData();
        formData.append('image', imageRef.current);

        const response = await fetch("http://localhost:8000/web/detect_recognize", {
          method: "POST",
          body: formData,
        });

        if (response.status === 200) {
          const text = await response.text();
          setResult(text);
        } else {
          setResult("Error from API.");
        }
      }
    }, 1000/FPS);
    return () => clearInterval(interval);
  }, []);

  const playCameraStream = () => {
    if (videoRef.current) {
      videoRef.current.play();
    }
  };

  const captureImageFromCamera = () => {
    const context = canvasRef.current.getContext('2d');
    const { videoWidth, videoHeight } = videoRef.current;

    canvasRef.current.width = videoWidth;
    canvasRef.current.height = videoHeight;

    context.drawImage(videoRef.current, 0, 0, videoWidth, videoHeight);

    canvasRef.current.toBlob((blob) => {
      imageRef.current = blob;
    })
  };

  return (
    <>
      <header>
        <h1>Image classifier</h1>
      </header>
      <main>
        <video ref={videoRef} onCanPlay={() => playCameraStream()} id="video" />
        <canvas ref={canvasRef} hidden></canvas>
        <p>Currently seeing: {result}</p>
      </main>
    </>
  )
};

export default Recognition;