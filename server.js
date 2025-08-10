const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Set view engine
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'templates'));

// Mock data and utilities for Indian stock market
const mockStockData = {
  'RELIANCE': {
    price: 2450.50,
    change: 25.30,
    changePercent: 1.04,
    volume: 1250000,
    marketCap: 1650000000000
  },
  'TCS': {
    price: 3680.75,
    change: -15.25,
    changePercent: -0.41,
    volume: 890000,
    marketCap: 1340000000000
  },
  'INFY': {
    price: 1456.20,
    change: 12.80,
    changePercent: 0.89,
    volume: 2100000,
    marketCap: 602000000000
  }
};

const getMarketStatus = () => {
  const now = new Date();
  const istTime = new Date(now.toLocaleString("en-US", {timeZone: "Asia/Kolkata"}));
  const hour = istTime.getHours();
  const minute = istTime.getMinutes();
  const currentTime = hour * 100 + minute;
  
  // Market hours: 9:15 AM to 3:30 PM IST
  const marketOpen = 915;
  const marketClose = 1530;
  
  if (currentTime >= marketOpen && currentTime <= marketClose) {
    return { status: 'open', message: 'Market is open' };
  } else {
    return { status: 'closed', message: 'Market is closed' };
  }
};

const formatIndianCurrency = (amount) => {
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR'
  }).format(amount);
};

// Routes
app.get('/', (req, res) => {
  res.render('index', { title: 'Indian Stock Technical Analysis' });
});

app.get('/api/analyze/:symbol', (req, res) => {
  try {
    const symbol = req.params.symbol.toUpperCase();
    const stockData = mockStockData[symbol];
    
    if (!stockData) {
      return res.status(404).json({ error: 'Stock not found' });
    }
    
    const marketStatus = getMarketStatus();
    
    // Mock technical analysis
    const analysis = {
      rsi: Math.random() * 100,
      macd: {
        macd: Math.random() * 10 - 5,
        signal: Math.random() * 10 - 5,
        histogram: Math.random() * 5 - 2.5
      },
      bollinger: {
        upper: stockData.price * 1.02,
        middle: stockData.price,
        lower: stockData.price * 0.98
      },
      support: stockData.price * 0.95,
      resistance: stockData.price * 1.05
    };
    
    // Mock trading signals
    const signals = {
      trend: Math.random() > 0.5 ? 'bullish' : 'bearish',
      strength: Math.random() * 100,
      recommendation: Math.random() > 0.6 ? 'buy' : Math.random() > 0.3 ? 'hold' : 'sell'
    };
    
    res.json({
      symbol,
      marketStatus,
      stockData,
      analysis,
      signals,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error(`Error analyzing ${req.params.symbol}:`, error);
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/sector-heatmap', (req, res) => {
  try {
    const sectors = [
      { name: 'Banking', change: 1.2, stocks: ['HDFC', 'ICICI', 'SBI'] },
      { name: 'IT', change: -0.8, stocks: ['TCS', 'INFY', 'WIPRO'] },
      { name: 'Energy', change: 2.1, stocks: ['RELIANCE', 'ONGC', 'IOC'] },
      { name: 'Pharma', change: 0.5, stocks: ['SUNPHARMA', 'DRREDDY', 'CIPLA'] },
      { name: 'Auto', change: -1.3, stocks: ['MARUTI', 'TATAMOTORS', 'M&M'] }
    ];
    
    res.json({ sectors, timestamp: new Date().toISOString() });
  } catch (error) {
    console.error('Error generating sector heatmap:', error);
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/orb-scanner', (req, res) => {
  try {
    const orbStocks = [
      { symbol: 'RELIANCE', breakoutPrice: 2475, currentPrice: 2480, volume: 1500000 },
      { symbol: 'TCS', breakoutPrice: 3700, currentPrice: 3695, volume: 950000 },
      { symbol: 'INFY', breakoutPrice: 1470, currentPrice: 1465, volume: 2200000 }
    ];
    
    res.json({ stocks: orbStocks, timestamp: new Date().toISOString() });
  } catch (error) {
    console.error('Error in ORB scanner:', error);
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/fii-dii-activity', (req, res) => {
  try {
    const activityData = {
      fii: {
        equity: Math.random() * 2000 - 1000, // Random between -1000 to 1000 crores
        debt: Math.random() * 500 - 250,
        hybrid: Math.random() * 100 - 50
      },
      dii: {
        equity: Math.random() * 1500 - 750,
        debt: Math.random() * 300 - 150,
        hybrid: Math.random() * 80 - 40
      },
      date: new Date().toISOString().split('T')[0]
    };
    
    res.json(activityData);
  } catch (error) {
    console.error('Error fetching FII/DII data:', error);
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/option-chain/:symbol', (req, res) => {
  try {
    const symbol = req.params.symbol.toUpperCase();
    const basePrice = mockStockData[symbol]?.price || 2500;
    
    const optionData = {
      symbol,
      underlyingPrice: basePrice,
      pcr: Math.random() * 2, // Put-Call Ratio
      maxPain: basePrice * (0.98 + Math.random() * 0.04),
      calls: [
        { strike: basePrice - 100, oi: Math.floor(Math.random() * 100000), volume: Math.floor(Math.random() * 50000) },
        { strike: basePrice - 50, oi: Math.floor(Math.random() * 150000), volume: Math.floor(Math.random() * 75000) },
        { strike: basePrice, oi: Math.floor(Math.random() * 200000), volume: Math.floor(Math.random() * 100000) },
        { strike: basePrice + 50, oi: Math.floor(Math.random() * 150000), volume: Math.floor(Math.random() * 75000) },
        { strike: basePrice + 100, oi: Math.floor(Math.random() * 100000), volume: Math.floor(Math.random() * 50000) }
      ],
      puts: [
        { strike: basePrice - 100, oi: Math.floor(Math.random() * 80000), volume: Math.floor(Math.random() * 40000) },
        { strike: basePrice - 50, oi: Math.floor(Math.random() * 120000), volume: Math.floor(Math.random() * 60000) },
        { strike: basePrice, oi: Math.floor(Math.random() * 180000), volume: Math.floor(Math.random() * 90000) },
        { strike: basePrice + 50, oi: Math.floor(Math.random() * 120000), volume: Math.floor(Math.random() * 60000) },
        { strike: basePrice + 100, oi: Math.floor(Math.random() * 80000), volume: Math.floor(Math.random() * 40000) }
      ]
    };
    
    res.json(optionData);
  } catch (error) {
    console.error(`Error fetching option chain for ${req.params.symbol}:`, error);
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/bulk-deals', (req, res) => {
  try {
    const dealsData = {
      bulkDeals: [
        { symbol: 'RELIANCE', buyer: 'ABC Investments', seller: 'XYZ Fund', quantity: 500000, price: 2450 },
        { symbol: 'TCS', buyer: 'DEF Capital', seller: 'GHI Holdings', quantity: 250000, price: 3680 }
      ],
      blockDeals: [
        { symbol: 'INFY', buyer: 'JKL Ventures', seller: 'MNO Trust', quantity: 1000000, price: 1456 }
      ],
      date: new Date().toISOString().split('T')[0]
    };
    
    res.json(dealsData);
  } catch (error) {
    console.error('Error fetching bulk deals:', error);
    res.status(500).json({ error: error.message });
  }
});

// Start server
app.listen(PORT, () => {
  const marketStatus = getMarketStatus();
  console.log(`Server running on port ${PORT}`);
  console.log(`Market status: ${marketStatus.message}`);
});