import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';

ReactDOM.render(
  <React.StrictMode>
    {/* 
    <div class="h-24 flex items-center shadow-md">
      <h1 class="text-5xl font-bold text-green-400 px-24">
        Boomer Humour Exhumer
      </h1>
    </div>
*/}

    <div class="flex justify-center bg-gray-800 h-screen">
      <div class="m-6 h-full">
        <div class="p-6 h-full">
          <h1 class="text-5xl font-bold text-green-400 py-3">
            Boomer Humour Exhumer
          </h1>
          <h2 class="text-4xl font-bold pt-3 pb-6 text-gray-300">
            Please upload an image
          </h2>
          <div class="border flex justify-center h-2/3 border-gray-300 rounded-lg items-center text-green-300 cursor-pointer hover:bg-green-300 hover:text-gray-800 hover:border-gray-800 py-3">
            <svg class="w-8 h-8" fill="currentColor" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
              <path d="M16.88 9.1A4 4 0 0 1 16 17H5a5 5 0 0 1-1-9.9V7a3 3 0 0 1 4.52-2.59A4.98 4.98 0 0 1 17 8c0 .38-.04.74-.12 1.1zM11 11h3l-4-4-4 4h3v3h2v-3z" />
            </svg>
            <span class="pl-2">Upload</span>
          </div>
        </div>
      </div>
    </div>
  </React.StrictMode>,
  document.getElementById('root')
);