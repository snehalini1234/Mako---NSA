import React, { useState } from 'react';
import { MapPin, Waves, Thermometer, Eye, Upload, TrendingUp, AlertCircle } from 'lucide-react';

// The main component, exported as default.
const SharkMonitoringDashboard = () => {
  const [activeTab, setActiveTab] = useState('map');
  const [selectedShark, setSelectedShark] = useState(null);
  const [sightings, setSightings] = useState([]);
  const [showUploadForm, setShowUploadForm] = useState(false);
  const [formData, setFormData] = useState({
    species: '',
    location: '',
    date: '',
    observer: ''
  });

  // Mock data for tracking
  const sharkData = [
    { id: 1, species: 'Great White', lat: -34.0, lon: 18.5, temp: 18.5, lastSeen: '2 hours ago', confidence: 92 },
    { id: 2, species: 'Tiger Shark', lat: -33.8, lon: 18.9, temp: 19.2, lastSeen: '5 hours ago', confidence: 87 },
    { id: 3, species: 'Bull Shark', lat: -33.5, lon: 18.3, temp: 20.1, lastSeen: '1 hour ago', confidence: 95 },
    { id: 4, species: 'Hammerhead', lat: -34.2, lon: 18.7, temp: 18.8, lastSeen: '3 hours ago', confidence: 89 },
  ];

  // Mock data for analytics
  const migrationRoutes = [
    { name: 'South African Coast', activity: 'high', sharks: 23 },
    { name: 'Mediterranean Basin', activity: 'medium', sharks: 15 },
    { name: 'Great Barrier Reef', activity: 'high', sharks: 31 },
    { name: 'California Coast', activity: 'low', sharks: 8 },
  ];

  // Handle form input changes
  const handleInputChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  // Handle community sighting submission
  const handleSightingSubmit = () => {
    if (formData.species && formData.location && formData.date && formData.observer) {
      const newSighting = {
        id: Date.now(),
        ...formData,
        status: 'pending' // Set status to pending validation
      };
      setSightings([newSighting, ...sightings]);
      setShowUploadForm(false);
      setFormData({ species: '', location: '', date: '', observer: '' });
    }
  };

  return (
    <div className="min-h-screen font-sans bg-gradient-to-br from-blue-900 via-blue-800 to-cyan-900 text-white">
      {/* Header Section */}
      <header className="sticky top-0 z-20 bg-black bg-opacity-30 backdrop-blur-md border-b border-cyan-500 border-opacity-30">
        <div className="container mx-auto px-4 sm:px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Waves className="w-8 h-8 sm:w-10 sm:h-10 text-cyan-400" />
              <div>
                <h1 className="text-xl sm:text-2xl font-bold bg-gradient-to-r from-cyan-300 to-blue-300 bg-clip-text text-transparent">
                  Sharks from Space
                </h1>
                <p className="text-xs sm:text-sm text-cyan-300 opacity-80">Satellite-Based Shark Monitoring System</p>
              </div>
            </div>
            <div className="hidden sm:block text-right">
              <div className="text-xs text-cyan-300">Last Data Fetch</div>
              <div className="text-sm font-semibold">2 min ago</div>
            </div>
          </div>
        </div>
      </header>

      {/* Tabs / Navigation */}
      <div className="container mx-auto px-4 sm:px-6 py-4">
        <div className="flex space-x-2 bg-black bg-opacity-30 rounded-xl p-1 shadow-inner shadow-cyan-900/50">
          {['map', 'analytics', 'sightings'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`flex-1 px-3 sm:px-6 py-2 sm:py-3 rounded-xl text-sm sm:text-base font-medium transition-all duration-300 ease-in-out ${
                activeTab === tab
                  ? 'bg-cyan-500 text-white shadow-lg shadow-cyan-500/50 transform scale-105'
                  : 'text-cyan-300 hover:bg-white hover:bg-opacity-10'
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content Area */}
      <div className="container mx-auto px-4 sm:px-6 pb-8">
        {/* Map View */}
        {activeTab === 'map' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 bg-black bg-opacity-40 backdrop-blur-md rounded-3xl p-6 border border-cyan-500 border-opacity-30 shadow-2xl shadow-cyan-900/40">
              <h2 className="text-xl font-bold mb-4 flex items-center">
                <MapPin className="w-6 h-6 mr-2 text-cyan-400" />
                Live Shark Tracking Map
              </h2>
              {/* Mock Map Container */}
              <div className="bg-gradient-to-br from-blue-950 to-cyan-950 rounded-2xl h-96 relative overflow-hidden border border-cyan-700/50">
                {/* Background Grid Pattern */}
                <div className="absolute inset-0 opacity-20">
                  <svg className="w-full h-full">
                    <defs>
                      <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                        <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#06b6d4" strokeWidth="0.5"/>
                      </pattern>
                    </defs>
                    <rect width="100%" height="100%" fill="url(#grid)" />
                  </svg>
                </div>
                
                {/* Shark Markers - Position based on Lat/Lon */}
                {sharkData.map((shark) => (
                  <div
                    key={shark.id}
                    className="absolute cursor-pointer transform -translate-x-1/2 -translate-y-1/2 group"
                    // FIX: Correctly use template literals inside the style object for CSS string values
                    style={{
                      left: `${((shark.lon + 180) / 360) * 100}%`,
                      top: `${((90 - shark.lat) / 180) * 100}%`,
                    }}
                    onClick={() => setSelectedShark(shark)}
                  >
                    <div className="relative">
                      {/* Pulsing effect */}
                      <div className="absolute inset-0 bg-cyan-400 rounded-full animate-ping opacity-75" />
                      {/* Main Marker */}
                      <div className="relative bg-cyan-500 rounded-full p-2 border-2 border-white shadow-xl shadow-cyan-500/70 transition-transform duration-150 group-hover:scale-125">
                        <Waves className="w-4 h-4" />
                      </div>
                    </div>
                    {/* Tooltip/Label */}
                    <div className="absolute top-full mt-2 left-1/2 transform -translate-x-1/2 bg-black bg-opacity-90 px-3 py-2 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300 whitespace-nowrap z-10 text-center">
                      <div className="text-sm font-semibold">{shark.species}</div>
                      <div className="text-xs text-cyan-300">{shark.lastSeen}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Side Panel: Detected Sharks & Environmental Data */}
            <div className="space-y-6">
              <div className="bg-black bg-opacity-40 backdrop-blur-md rounded-3xl p-6 border border-cyan-500 border-opacity-30 shadow-xl">
                <h3 className="text-lg font-bold mb-4">Detected Sharks</h3>
                <div className="space-y-3">
                  {sharkData.map((shark) => (
                    <div
                      key={shark.id}
                      onClick={() => setSelectedShark(shark)}
                      className={`p-4 rounded-xl cursor-pointer transition-all duration-200 ease-in-out border-2 ${
                        selectedShark?.id === shark.id
                          ? 'bg-cyan-500 bg-opacity-30 border-cyan-400 shadow-md shadow-cyan-500/30'
                          : 'bg-white bg-opacity-5 hover:bg-opacity-10 border-transparent hover:border-cyan-700'
                      }`}
                    >
                      <div className="flex justify-between items-start mb-2">
                        <div className="font-semibold">{shark.species}</div>
                        <div className="text-xs bg-cyan-500 px-3 py-1 rounded-full font-bold">
                          {shark.confidence}%
                        </div>
                      </div>
                      <div className="text-sm space-y-1">
                        <div className="flex items-center text-cyan-300">
                          <MapPin className="w-3 h-3 mr-2 text-cyan-400" />
                          {shark.lat.toFixed(2)}°, {shark.lon.toFixed(2)}°
                        </div>
                        <div className="flex items-center text-cyan-300">
                          <Thermometer className="w-3 h-3 mr-2 text-cyan-400" />
                          {shark.temp}°C
                        </div>
                        <div className="text-xs opacity-70 mt-1">{shark.lastSeen}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-black bg-opacity-40 backdrop-blur-md rounded-3xl p-6 border border-cyan-500 border-opacity-30 shadow-xl">
                <h3 className="text-lg font-bold mb-4">Environmental Data</h3>
                <div className="space-y-4">
                  {/* SST Bar */}
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span>Sea Surface Temp</span>
                      <span className="font-semibold text-cyan-300">18.5°C</span>
                    </div>
                    <div className="h-3 bg-black bg-opacity-50 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-blue-500 to-cyan-400 transition-all duration-1000" style={{width: '65%'}} />
                    </div>
                  </div>
                  {/* Clarity Bar */}
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span>Water Clarity</span>
                      <span className="font-semibold text-cyan-300">High</span>
                    </div>
                    <div className="h-3 bg-black bg-opacity-50 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-cyan-400 to-green-400 transition-all duration-1000" style={{width: '85%'}} />
                    </div>
                  </div>
                  {/* Chlorophyll Bar */}
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span>Chlorophyll Level</span>
                      <span className="font-semibold text-cyan-300">Moderate</span>
                    </div>
                    <div className="h-3 bg-black bg-opacity-50 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-yellow-400 to-red-400 transition-all duration-1000" style={{width: '55%'}} />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Analytics View */}
        {activeTab === 'analytics' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-black bg-opacity-40 backdrop-blur-md rounded-3xl p-6 border border-cyan-500 border-opacity-30 shadow-2xl shadow-cyan-900/40">
              <h2 className="text-xl font-bold mb-6 flex items-center">
                <TrendingUp className="w-6 h-6 mr-2 text-cyan-400" />
                Migration Hotspots
              </h2>
              <div className="space-y-4">
                {migrationRoutes.map((route, idx) => (
                  <div key={idx} className="p-4 bg-white bg-opacity-5 rounded-xl border border-cyan-900 hover:bg-opacity-10 transition-all duration-200">
                    <div className="flex justify-between items-center mb-2">
                      <div className="font-semibold text-lg">{route.name}</div>
                      <div className={`px-3 py-1 rounded-full text-xs font-bold ${
                        route.activity === 'high' ? 'bg-red-600' :
                        route.activity === 'medium' ? 'bg-yellow-600' : 'bg-green-600'
                      }`}>
                        {route.activity.toUpperCase()}
                      </div>
                    </div>
                    <div className="flex items-center justify-between text-sm text-cyan-300">
                      <span>Detected Sharks: <span className="font-bold text-white">{route.sharks}</span></span>
                      <Waves className="w-4 h-4 text-cyan-400" />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="space-y-6">
              <div className="bg-black bg-opacity-40 backdrop-blur-md rounded-3xl p-6 border border-cyan-500 border-opacity-30 shadow-xl">
                <h2 className="text-xl font-bold mb-4">Detection Statistics</h2>
                <div className="grid grid-cols-2 gap-4">
                  <StatCard title="Total Detections" value="77" gradient="from-cyan-500 to-blue-500" />
                  <StatCard title="Species Tracked" value="12" gradient="from-blue-500 to-purple-500" />
                  <StatCard title="Avg Confidence" value="91%" gradient="from-purple-500 to-pink-500" />
                  <StatCard title="Community Reports" value="45" gradient="from-pink-500 to-red-500" />
                </div>
              </div>

              <div className="bg-black bg-opacity-40 backdrop-blur-md rounded-3xl p-6 border border-cyan-500 border-opacity-30 shadow-xl">
                <h3 className="text-lg font-bold mb-4 flex items-center">
                  <AlertCircle className="w-5 h-5 mr-2 text-yellow-400" />
                  Conservation Alerts
                </h3>
                <div className="space-y-3">
                  <AlertItem 
                    title="Increased Activity" 
                    subtitle="South African Coast - Great White sharks" 
                    color="yellow" 
                  />
                  <AlertItem 
                    title="Migration Pattern Shift" 
                    subtitle="Tiger sharks moving north early due to temperature change" 
                    color="blue" 
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Sightings View */}
        {activeTab === 'sightings' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <div className="bg-black bg-opacity-40 backdrop-blur-md rounded-3xl p-6 border border-cyan-500 border-opacity-30 shadow-2xl shadow-cyan-900/40">
                <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6">
                  <h2 className="text-xl font-bold flex items-center mb-4 sm:mb-0">
                    <Eye className="w-6 h-6 mr-2 text-cyan-400" />
                    Community Sightings
                  </h2>
                  <button
                    onClick={() => setShowUploadForm(!showUploadForm)}
                    className="flex items-center space-x-2 px-4 py-2 bg-cyan-500 hover:bg-cyan-600 rounded-xl transition-colors font-semibold shadow-md shadow-cyan-500/50"
                  >
                    <Upload className="w-4 h-4" />
                    <span>Report Sighting</span>
                  </button>
                </div>

                {/* Sighting Upload Form */}
                {showUploadForm && (
                  <div className="mb-6 p-6 bg-white bg-opacity-5 rounded-xl space-y-4 border border-cyan-500/30">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <InputField 
                        label="Species" 
                        placeholder="e.g., Great White" 
                        value={formData.species} 
                        onChange={(e) => handleInputChange('species', e.target.value)} 
                      />
                      <InputField 
                        label="Location (GPS/Name)" 
                        placeholder="Coordinates or location name" 
                        value={formData.location} 
                        onChange={(e) => handleInputChange('location', e.target.value)} 
                      />
                      <InputField 
                        label="Date and Time" 
                        placeholder="e.g., 2025-10-02 14:30" 
                        value={formData.date} 
                        onChange={(e) => handleInputChange('date', e.target.value)} 
                      />
                      <InputField 
                        label="Your Name/Handle" 
                        placeholder="Observer name" 
                        value={formData.observer} 
                        onChange={(e) => handleInputChange('observer', e.target.value)} 
                      />
                    </div>
                    <button
                      onClick={handleSightingSubmit}
                      className="w-full py-3 bg-cyan-500 hover:bg-cyan-600 rounded-xl font-bold transition-colors shadow-lg shadow-cyan-500/30 mt-2"
                    >
                      Submit Sighting for Validation
                    </button>
                  </div>
                )}

                {/* Sighting List */}
                <div className="space-y-4">
                  {sightings.length === 0 ? (
                    <div className="text-center py-12 text-cyan-300 opacity-60 bg-white bg-opacity-5 rounded-xl border border-dashed border-cyan-700/50">
                      <Eye className="w-8 h-8 mx-auto mb-2" />
                      <p>No community sightings yet. Be the first to report!</p>
                    </div>
                  ) : (
                    sightings.map((sighting) => (
                      <SightingCard key={sighting.id} sighting={sighting} />
                    ))
                  )}
                </div>
              </div>
            </div>

            {/* How to Report Panel */}
            <div className="bg-black bg-opacity-40 backdrop-blur-md rounded-3xl p-6 border border-cyan-500 border-opacity-30 shadow-xl">
              <h3 className="text-lg font-bold mb-4">How to Report</h3>
              <div className="space-y-4 text-sm text-cyan-300">
                <p className="text-white text-base">Help us track sharks and support conservation efforts by submitting confirmed sightings.</p>
                <StepItem step="1" text="Note the exact location (GPS preferred) and time of your sighting." />
                <StepItem step="2" text="Identify the species and estimate size if possible." />
                <StepItem step="3" text="Submit your report using the form above (photos/videos highly encouraged)." />
                <StepItem step="4" text="Our AI validation engine will verify the sighting status." />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// --- Sub-Components for cleaner JSX ---

const StatCard = ({ title, value, gradient }) => (
    <div className={`p-4 rounded-xl bg-gradient-to-br ${gradient} shadow-lg`}>
        <div className="text-3xl font-extrabold">{value}</div>
        <div className="text-sm opacity-90">{title}</div>
    </div>
);

const AlertItem = ({ title, subtitle, color }) => (
    <div className={`p-3 bg-${color}-500 bg-opacity-20 border-l-4 border-${color}-400 rounded`}>
        <div className="font-semibold text-sm">{title}</div>
        <div className="text-xs opacity-80">{subtitle}</div>
    </div>
);

const InputField = ({ label, placeholder, value, onChange }) => (
    <div>
        <label className="block text-sm font-medium mb-2">{label}</label>
        <input
            type="text"
            value={value}
            onChange={onChange}
            className="w-full px-4 py-2 bg-black bg-opacity-50 border border-cyan-500 border-opacity-30 rounded-xl focus:outline-none focus:border-cyan-400 text-white placeholder-cyan-200/50"
            placeholder={placeholder}
        />
    </div>
);

const SightingCard = ({ sighting }) => {
    // Map status to visual styles
    const statusClasses = {
        pending: 'bg-yellow-500 text-yellow-200 border-yellow-500',
        validated: 'bg-green-500 text-green-200 border-green-500',
        rejected: 'bg-red-500 text-red-200 border-red-500',
    };
    
    // Default to pending if status is missing
    const status = sighting.status || 'pending';

    return (
        <div className="p-4 bg-white bg-opacity-5 rounded-xl hover:bg-opacity-10 transition-all duration-200 border border-transparent hover:border-cyan-700/50">
            <div className="flex justify-between items-start">
                <div>
                    <div className="font-semibold text-lg">{sighting.species}</div>
                    <div className="text-sm text-cyan-300 mt-0.5"><MapPin className="inline w-3 h-3 mr-1" /> {sighting.location}</div>
                    <div className="text-xs opacity-70 mt-2">Reported by: <span className="font-medium text-white">{sighting.observer}</span></div>
                    <div className="text-xs opacity-70">{sighting.date}</div>
                </div>
                <div 
                    className={`px-3 py-1 rounded-full text-xs font-bold border ${statusClasses[status]} bg-opacity-20`}
                >
                    {status.toUpperCase()}
                </div>
            </div>
        </div>
    );
};

const StepItem = ({ step, text }) => (
  <div className="flex items-start">
    <div className="w-6 h-6 bg-cyan-500 rounded-full flex items-center justify-center mr-3 mt-0.5 flex-shrink-0 font-bold text-black text-sm">{step}</div>
    <div className='text-sm'>{text}</div>
  </div>
);

export default SharkMonitoringDashboard;