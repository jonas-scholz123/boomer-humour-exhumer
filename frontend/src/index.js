import React, { useState } from 'react';
import ReactDOM from 'react-dom';
import './index.css';


const ImageUploadCard = () => {
  const [image, setImage] = useState(null);

  const onImageChange = event => {
    if (event.target.files && event.target.files[0]) {
      setImage(event.target.files[0])
    }
  };

  const UploadButton = () => {
    return (
      <div class="flex items-center justify-center">
        <svg class="w-8 h-8" fill="currentColor" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
          <path d="M16.88 9.1A4 4 0 0 1 16 17H5a5 5 0 0 1-1-9.9V7a3 3 0 0 1 4.52-2.59A4.98 4.98 0 0 1 17 8c0 .38-.04.74-.12 1.1zM11 11h3l-4-4-4 4h3v3h2v-3z" />
        </svg>
        <span class="pl-2">Upload</span>
      </div>
    )
  }

  const UploadedImage = () => {
    return (
      <div class="h-full w-full relative z-0 group">
        <div className="w-full h-full z-10 absolute flex justify-center items-center opacity-0 hover:opacity-100 group-hover:text-green-300 ">
          <svg class="w-8 h-8 px-1" fill="currentColor" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
            <path d="M16.88 9.1A4 4 0 0 1 16 17H5a5 5 0 0 1-1-9.9V7a3 3 0 0 1 4.52-2.59A4.98 4.98 0 0 1 17 8c0 .38-.04.74-.12 1.1zM11 11h3l-4-4-4 4h3v3h2v-3z" />
          </svg>
          <p className="text-2xl font-semibold">Replace Image</p>
        </div>
        <img src={URL.createObjectURL(image)} class="h-full w-full absolute rounded-lg border inset-0 border-green-300 group-hover:filter group-hover:grayscale group-hover:blur-sm group-hover:brightness-90"/>
      </div>
    )
  }

  const Exhume = () => {
    const data = new FormData();

    data.append('image', image);
    data.append('filename', "test");

    fetch('http://localhost:5000/api/exhume', {
      method: "POST",
      body: data,
    });

  }

  const ExhumeButton = () => {
    return (
      <div class="flex justify-center items-center w-full py-6">
        <button class="rounded-md bg-gray-800 h-12 w-full border-green-300 hover:bg-green-300 text-green-300 font-semibold hover:text-gray-800 py-2 px-4 border hover:border-transparent rounded" onClick={() => Exhume()}>
          Exhume
        </button>
      </div>
    )
  }


  return (
    <div class="flex justify-center bg-gray-800 h-screen">
      <div class="m-6 h-full">
        <div class="p-6 h-full">
          <h1 class="text-5xl font-bold text-green-400 py-3">
            Boomer Humour Exhumer
          </h1>
          <h2 class="text-4xl font-bold pt-3 pb-6 text-gray-300">
            Please upload an image
          </h2>


          <div class="h-full">
            <div class="invisible">
              <form encType="multipart/form-data">
              <input
                type="file"
                name="myImage"
                id="fileUpload"
                onChange={event => onImageChange(event)}
              >
              </input>
              </form>


            </div>
            <label
              for="fileUpload"
              class="h-1/2 border flex justify-center border-gray-300 rounded-lg items-center text-green-300 cursor-pointer hover:bg-green-300 hover:text-gray-800 hover:border-gray-800"
            >

              {image === null && <UploadButton/>}
              {image !== null && <UploadedImage/>}

            </label>

            {image !== null && <ExhumeButton />}

          </div>
        </div>
      </div>
    </div>

  )
}


ReactDOM.render(
  <React.StrictMode>
    {/* 
    <div class="h-24 flex items-center shadow-md">
      <h1 class="text-5xl font-bold text-green-400 px-24">
        Boomer Humour Exhumer
      </h1>
    </div>
*/}
    <ImageUploadCard/>

  </React.StrictMode>,
  document.getElementById('root')
);
