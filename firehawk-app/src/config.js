// frontend/src/config.js

// Check if the site is running on your computer
const isLocal = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";

const config = {
  // USER BACKEND URL (Node.js/Express - Login, Register)
  // On Railway, put your Auth project URL here
  AUTH_API: isLocal 
    ? "http://localhost:5000" 
    : "https://firehawk-project-production.up.railway.app", 

  // PYTHON PIPELINE URL (Data, Weather, AI)
  // Flask defined on port 5000 - adjust if needed
  // On Railway, put the Python project URL here
  //DATA_API: isLocal 
    //? "http://localhost:5000" 
    //: "https://firehawk-pipeline-production.up.railway.app",
};

export default config;