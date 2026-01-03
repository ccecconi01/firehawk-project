const express = require('express');
const cors = require('cors');
require('dotenv').config();
const loginRoute = require('./loginRoute');

const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// Routes
app.use('/api', loginRoute);

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ status: 'Server is running' });
});

// Error handling
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({
    success: false,
    message: 'Internal server error',
    error: err.message
  });
});

// Start server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`FireHawk Backend running on http://localhost:${PORT}`  );
  console.log(`Database: ${process.env.DB_NAME}`);
});