// server.js - Main Express server for Sharks from Space
const express = require('express');
const cors = require('cors');
const mongoose = require('mongoose');
const multer = require('multer');
const path = require('path');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());
app.use('/uploads', express.static('uploads'));

// MongoDB Connection
mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/sharks_from_space', {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

const db = mongoose.connection;
db.on('error', console.error.bind(console, 'MongoDB connection error:'));
db.once('open', () => {
  console.log('Connected to MongoDB');
});

// Mongoose Schemas
const sharkDetectionSchema = new mongoose.Schema({
  species: { type: String, required: true },
  latitude: { type: Number, required: true },
  longitude: { type: Number, required: true },
  confidence: { type: Number, required: true },
  temperature: Number,
  depth: Number,
  satelliteSource: String,
  imageUrl: String,
  detectedAt: { type: Date, default: Date.now },
  environmentalData: {
    seaSurfaceTemp: Number,
    chlorophyllLevel: Number,
    waterClarity: Number,
    salinity: Number
  },
  metadata: {
    detectionMethod: String,
    modelVersion: String,
    processedBy: String
  }
});

const communitySightingSchema = new mongoose.Schema({
  species: { type: String, required: true },
  location: {
    type: { type: String, default: 'Point' },
    coordinates: [Number], // [longitude, latitude]
    description: String
  },
  observerName: { type: String, required: true },
  observerEmail: String,
  observerType: { type: String, enum: ['diver', 'boater', 'researcher', 'other'], default: 'other' },
  sightingDate: { type: Date, required: true },
  reportedAt: { type: Date, default: Date.now },
  status: { type: String, enum: ['pending', 'verified', 'rejected'], default: 'pending' },
  verifiedBy: String,
  verifiedAt: Date,
  notes: String,
  photoUrl: String,
  additionalInfo: {
    sharkSize: String,
    behavior: String,
    numberOfSharks: Number,
    waterConditions: String
  }
});

const migrationRouteSchema = new mongoose.Schema({
  name: { type: String, required: true },
  region: String,
  coordinates: [[Number]], // Array of [longitude, latitude] pairs
  species: [String],
  activityLevel: { type: String, enum: ['low', 'medium', 'high'], default: 'medium' },
  sharkCount: { type: Number, default: 0 },
  season: String,
  lastUpdated: { type: Date, default: Date.now }
});

// Create indexes for geospatial queries
communitySightingSchema.index({ location: '2dsphere' });

// Models
const SharkDetection = mongoose.model('SharkDetection', sharkDetectionSchema);
const CommunitySighting = mongoose.model('CommunitySighting', communitySightingSchema);
const MigrationRoute = mongoose.model('MigrationRoute', migrationRouteSchema);

// Multer configuration for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + path.extname(file.originalname));
  }
});

const upload = multer({ 
  storage: storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB limit
  fileFilter: (req, file, cb) => {
    const filetypes = /jpeg|jpg|png|gif/;
    const mimetype = filetypes.test(file.mimetype);
    const extname = filetypes.test(path.extname(file.originalname).toLowerCase());
    
    if (mimetype && extname) {
      return cb(null, true);
    }
    cb(new Error('Only image files are allowed'));
  }
});

// Routes

// Get all shark detections
app.get('/api/detections', async (req, res) => {
  try {
    const { species, startDate, endDate, minConfidence, limit = 100 } = req.query;
    
    let query = {};
    
    if (species) {
      query.species = new RegExp(species, 'i');
    }
    
    if (startDate || endDate) {
      query.detectedAt = {};
      if (startDate) query.detectedAt.$gte = new Date(startDate);
      if (endDate) query.detectedAt.$lte = new Date(endDate);
    }
    
    if (minConfidence) {
      query.confidence = { $gte: parseFloat(minConfidence) };
    }
    
    const detections = await SharkDetection.find(query)
      .sort({ detectedAt: -1 })
      .limit(parseInt(limit));
    
    res.json({
      success: true,
      count: detections.length,
      data: detections
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Get detections within a bounding box
app.get('/api/detections/area', async (req, res) => {
  try {
    const { minLat, maxLat, minLon, maxLon } = req.query;
    
    if (!minLat || !maxLat || !minLon || !maxLon) {
      return res.status(400).json({ 
        success: false, 
        error: 'Missing bounding box parameters' 
      });
    }
    
    const detections = await SharkDetection.find({
      latitude: { $gte: parseFloat(minLat), $lte: parseFloat(maxLat) },
      longitude: { $gte: parseFloat(minLon), $lte: parseFloat(maxLon) }
    }).sort({ detectedAt: -1 });
    
    res.json({
      success: true,
      count: detections.length,
      data: detections
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Create new shark detection (for satellite processing system)
app.post('/api/detections', async (req, res) => {
  try {
    const detection = new SharkDetection(req.body);
    await detection.save();
    
    res.status(201).json({
      success: true,
      data: detection
    });
  } catch (error) {
    res.status(400).json({ success: false, error: error.message });
  }
});

// Get all community sightings
app.get('/api/sightings', async (req, res) => {
  try {
    const { status, species, limit = 50 } = req.query;
    
    let query = {};
    
    if (status) {
      query.status = status;
    }
    
    if (species) {
      query.species = new RegExp(species, 'i');
    }
    
        const sightings = await CommunitySighting.find(query)
          .sort({ reportedAt: -1 })
          .limit(parseInt(limit));
    
        res.json({
          success: true,
          count: sightings.length,
          data: sightings
        });
      } catch (error) {
        res.status(500).json({ success: false, error: error.message });
      }
    });