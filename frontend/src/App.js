// import logo from './logo.svg';
import './App.css';
import {useEffect} from 'react'
import Recognition from './recognition';
function App() {
  var ws = null
  var image = null
  var blob
  // useEffect(() => {
  //   ws = new WebSocket("ws://localhost:8000/ws");
  //   ws.onopen = () => ws.send("Connected");
  //   onmeaage();
  // }, []);
  let video = document.querySelector('video'); 
  let blobArray = [];
  // socket.on('message',data=>{
  //   blobArray.push(new Blob([new Uint8Array(data)],{'type':'video/mp4'}));
  //   let currentTime = video.currentTime;
  //   let blob = new Blob(blobArray,{'type':'video/mp4'});
  //   video.src = window.URL.createObjectURL(blob);
  //   video.currentTime = currentTime;
  //   video.play();
  // });
  // const onmeaage = () => {
  //   ws.onmessage = (event) => {
  //     // blobArray.push(new Blob([new Uint8Array(event.data)],{'type':'video/mp4'}));
  //     // let blob = new Blob(event.data,{'type':'video/mp4'});
  //     // video.src = window.URL.createObjectURL(blob);
  //     // video.play()
  //     blob = event.data
  //     console.log("WebSocket message received:", event.data);
  //   };
  // };
  return (
    <div className="App">
      <Recognition />
      {/* <img src="{blob}"></img> */}
      {/* <video width="300" controls muted="muted">
        </video> */}
    </div>
  );
}

export default App;
