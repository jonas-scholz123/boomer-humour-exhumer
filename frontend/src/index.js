import React, { useState } from 'react';
import ReactDOM from 'react-dom';
import './index.css';



const ImageUploadCard = () => {
  const [image, setImage] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [boomerness, setBoomerness] = useState(null)

  const onImageChange = event => {
    if (event.target.files && event.target.files[0]) {
      setImage(event.target.files[0])
      setBoomerness(null)
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

  const BoomerText = () => {
    return (
      <div>
        <p class="text-3xl text-green-300 font-semibold">The image is {boomerness}% boomerish.</p>
      </div>
    )
  }

  const Exhume = () => {
    const data = new FormData();

    data.append('image', image);
    data.append('filename', "test");
    setLoading(true)
    fetch('http://localhost:5000/api/exhume', {
      method: "POST",
      body: data,
    })
    .then( response => response.json())
    .then( data => {
      console.log(data)
      setBoomerness(data.boomerness)
      setLoading(false)
    })
    .catch( error => {
      console.log(error)
      setError(error)
    })

  }

  const ExhumeButton = () => {
    if (boomerness === null){
      return (
        <div class="flex justify-center items-center w-full py-6">
          <button class="rounded-md bg-gray-800 h-12 w-full border-green-300 hover:bg-green-300 text-green-300 font-semibold hover:text-gray-800 py-2 px-4 border hover:border-transparent rounded" onClick={() => Exhume()}>
            Exhume
          </button>
        </div>
      )
    }
    return (
      <div className="relative py-6">
        <div className="overflow-hidden h-12 mb-4 text-xs flex rounded border border-green-300">
          <div style={{ width: `${boomerness}%`, transition: "width 2s" }} className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-green-300 transition-width transition-500 ease"></div>
        </div>
      </div>
      /*
      <div class="relative pt-1">
        <div class="overflow-hidden h-2 mb-4 text-xs flex rounded bg-pink-200">
          <div style="width:30%" class="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-pink-500"></div>
        </div>
      </div> */
    )
  }


  return (
    <div class="flex justify-center bg-gray-800 h-screen overflow-hidden">
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
            {boomerness !== null && <BoomerText/>}

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
