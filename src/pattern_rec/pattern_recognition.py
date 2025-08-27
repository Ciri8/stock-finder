"""
CNN-based pattern recognition for chart patterns.
Detects breakout patterns like flags, triangles, cup & handle, etc.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import cv2
from PIL import Image
import io
import logging
from dataclasses import dataclass
import os
from datetime import datetime
import json

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.data_splitter import DataSplitter

logger = logging.getLogger(__name__)


@dataclass
class PatternDetection:
    """Container for detected pattern information."""
    pattern_type: str
    confidence: float
    start_idx: int
    end_idx: int
    breakout_point: Optional[int] = None
    price_target: Optional[float] = None
    image_path: Optional[str] = None


class ChartPatternDataset(Dataset):
    """Dataset for chart pattern images."""
    
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return torch.FloatTensor(image), torch.LongTensor([label])


class PatternCNN(nn.Module):
    """CNN model for chart pattern recognition."""
    
    def __init__(self, num_patterns: int = 12, input_channels: int = 3):
        super(PatternCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # Fully connected layers
        # Assuming input image size of 224x224, after 4 pooling layers: 224/16 = 14
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_patterns)
        
        # Activation
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Conv Block 1
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        
        # Conv Block 2
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        # Conv Block 3
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # Conv Block 4
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x
    
    def predict_with_confidence(self, x):
        """Get predictions with confidence scores."""
        logits = self.forward(x)
        probs = self.softmax(logits)
        return probs


class ChartPatternRecognizer:
    """Main class for chart pattern recognition."""
    
    # Pattern definitions
    PATTERNS = {
        0: 'no_pattern',
        1: 'bull_flag',
        2: 'bear_flag',
        3: 'ascending_triangle',
        4: 'descending_triangle',
        5: 'symmetric_triangle',
        6: 'cup_and_handle',
        7: 'inverse_head_shoulders',
        8: 'head_shoulders',
        9: 'double_top',
        10: 'double_bottom',
        11: 'wedge'
    }
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = None,
                 image_size: Tuple[int, int] = (224, 224)):
        """
        Initialize pattern recognizer.
        
        Args:
            model_path: Path to saved model weights
            device: 'cuda' or 'cpu' (auto-detect if None)
            image_size: Size of chart images (width, height)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        self.image_size = image_size
        self.model = PatternCNN(num_patterns=len(self.PATTERNS)).to(self.device)
        
        # Load pre-trained weights if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5
        )
    
    def ohlcv_to_image(self, df: pd.DataFrame, window_size: int = 60) -> np.ndarray:
        """
        Convert OHLCV data to candlestick chart image.
        
        Args:
            df: DataFrame with OHLC data
            window_size: Number of candles to include
            
        Returns:
            Image array of shape (channels, height, width)
        """
        # Take last window_size rows
        df_window = df.tail(window_size).copy()
        
        # Prepare data for candlestick chart
        df_window.reset_index(inplace=True)
        if 'Date' in df_window.columns:
            df_window['Date'] = pd.to_datetime(df_window['Date'])
            df_window['Date'] = df_window['Date'].map(mdates.date2num)
        elif 'date' in df_window.columns:
            df_window['date'] = pd.to_datetime(df_window['date'])
            df_window['date'] = df_window['date'].map(mdates.date2num)
        else:
            df_window['date'] = range(len(df_window))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6), dpi=28)  # Results in 224x168 image
        
        # Ensure columns are lowercase
        df_window.columns = df_window.columns.str.lower()
        
        # Plot candlestick chart
        ohlc_data = df_window[['date', 'open', 'high', 'low', 'close']].values
        candlestick_ohlc(ax, ohlc_data, width=0.6, 
                         colorup='green', colordown='red', alpha=0.8)
        
        # Add volume subplot
        ax2 = ax.twinx()
        volume = df_window['volume'].values
        x = df_window['date'].values
        
        # Color volume bars based on price change
        colors = ['green' if df_window['close'].iloc[i] >= df_window['open'].iloc[i] 
                 else 'red' for i in range(len(df_window))]
        ax2.bar(x, volume, color=colors, alpha=0.3, width=0.6)
        
        # Add moving averages
        date_col = 'date' if 'date' in df_window.columns else 'Date'
        if len(df_window) >= 20:
            ma20 = df_window['close'].rolling(20).mean()
            ax.plot(df_window[date_col], ma20, 'b-', alpha=0.7, linewidth=1)
        
        if len(df_window) >= 50:
            ma50 = df_window['close'].rolling(50).mean()
            ax.plot(df_window[date_col], ma50, 'orange', alpha=0.7, linewidth=1)
        
        # Style the chart
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax2.set_ylabel('')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax2.set_yticklabels([])
        
        # Convert to image array
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close()
        
        # Load image and resize
        img = Image.open(buf).convert('RGB')
        img = img.resize(self.image_size, Image.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Transpose to (channels, height, width) for PyTorch
        img_array = img_array.transpose(2, 0, 1)
        
        return img_array
    
    def detect_patterns(self, df: pd.DataFrame, 
                       confidence_threshold: float = 0.7) -> List[PatternDetection]:
        """
        Detect patterns in the given price data.
        
        Args:
            df: DataFrame with OHLCV data
            confidence_threshold: Minimum confidence for pattern detection
            
        Returns:
            List of detected patterns
        """
        self.model.eval()
        detections = []
        
        # Generate chart image
        chart_image = self.ohlcv_to_image(df)
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.FloatTensor(chart_image).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            probs = self.model.predict_with_confidence(image_tensor)
            probs = probs.cpu().numpy()[0]
        
        # Get top predictions
        top_indices = np.argsort(probs)[::-1]
        
        for idx in top_indices[:3]:  # Top 3 patterns
            confidence = probs[idx]
            if confidence >= confidence_threshold and idx != 0:  # Skip 'no_pattern'
                pattern = PatternDetection(
                    pattern_type=self.PATTERNS[idx],
                    confidence=float(confidence),
                    start_idx=max(0, len(df) - 60),
                    end_idx=len(df) - 1,
                    breakout_point=self._find_breakout_point(df, self.PATTERNS[idx])
                )
                detections.append(pattern)
        
        return detections
    
    def _find_breakout_point(self, df: pd.DataFrame, pattern_type: str) -> Optional[int]:
        """
        Find the breakout point for a given pattern.
        
        Args:
            df: DataFrame with price data
            pattern_type: Type of pattern detected
            
        Returns:
            Index of breakout point or None
        """
        if len(df) < 20:
            return None
        
        # Simple breakout detection: price breaks above 20-day high
        recent_high = df['high'].tail(20).max()
        last_close = df['close'].iloc[-1]
        
        if last_close > recent_high * 0.99:  # Within 1% of breaking out
            return len(df) - 1
        
        return None
    
    def highlight_pattern_on_chart(self, df: pd.DataFrame, 
                                  pattern: PatternDetection,
                                  save_path: str) -> str:
        """
        Create and save a chart with the pattern highlighted.
        
        Args:
            df: DataFrame with OHLCV data
            pattern: Detected pattern information
            save_path: Directory to save the image
            
        Returns:
            Path to saved image
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data
        df_copy = df.copy()
        df_copy.reset_index(inplace=True)
        if 'Date' in df_copy.columns:
            df_copy['Date'] = pd.to_datetime(df_copy['Date'])
            df_copy['Date'] = df_copy['Date'].map(mdates.date2num)
        else:
            df_copy['Date'] = range(len(df_copy))
        
        # Ensure columns are properly cased
        df_copy.columns = df_copy.columns.str.lower()
        
        # Plot candlesticks
        ohlc_data = df_copy[['date', 'open', 'high', 'low', 'close']].values
        candlestick_ohlc(ax, ohlc_data, width=0.6, 
                         colorup='green', colordown='red', alpha=0.8)
        
        # Highlight pattern area
        start_x = df_copy['date'].iloc[pattern.start_idx]
        end_x = df_copy['date'].iloc[pattern.end_idx]
        y_min = df_copy['low'].iloc[pattern.start_idx:pattern.end_idx].min()
        y_max = df_copy['high'].iloc[pattern.start_idx:pattern.end_idx].max()
        
        # Add pattern rectangle
        rect = mpatches.Rectangle((start_x, y_min), end_x - start_x, y_max - y_min,
                                 fill=True, alpha=0.2, color='blue',
                                 label=f"{pattern.pattern_type} ({pattern.confidence:.1%})")
        ax.add_patch(rect)
        
        # Mark breakout point if exists
        if pattern.breakout_point:
            ax.scatter(df_copy['date'].iloc[pattern.breakout_point],
                      df_copy['high'].iloc[pattern.breakout_point],
                      color='red', s=100, marker='^', zorder=5,
                      label='Breakout Point')
        
        # Styling
        ax.set_title(f"Pattern Detection: {pattern.pattern_type}", fontsize=14)
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{pattern.pattern_type}_{timestamp}.png"
        filepath = os.path.join(save_path, filename)
        
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def train_model(self, train_loader: DataLoader, 
                   val_loader: DataLoader,
                   epochs: int = 50,
                   save_best: bool = True) -> Dict:
        """
        Train the CNN model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            save_best: Save best model based on validation loss
            
        Returns:
            Training history dictionary
        """
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.squeeze().to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.squeeze().to(self.device)
                    
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100 * correct / total
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            
            # Update learning rate
            self.scheduler.step(avg_val_loss)
            
            # Save best model
            if save_best and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model('best_pattern_model.pth')
            
            # Log progress
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch}/{epochs} - "
                          f"Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {avg_val_loss:.4f}, "
                          f"Val Acc: {val_acc:.2f}%")
        
        return history
    
    def save_model(self, path: str):
        """Save model weights."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'patterns': self.PATTERNS
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {path}")
    
    def batch_detect(self, stock_data: Dict[str, pd.DataFrame],
                    confidence_threshold: float = 0.7) -> Dict[str, List[PatternDetection]]:
        """
        Detect patterns for multiple stocks.
        
        Args:
            stock_data: Dictionary mapping ticker to DataFrame
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            Dictionary mapping ticker to list of detected patterns
        """
        results = {}
        
        for ticker, df in stock_data.items():
            try:
                patterns = self.detect_patterns(df, confidence_threshold)
                results[ticker] = patterns
                
                if patterns:
                    logger.info(f"{ticker}: Detected {len(patterns)} patterns")
                    for p in patterns:
                        logger.info(f"  - {p.pattern_type}: {p.confidence:.1%} confidence")
            except Exception as e:
                logger.error(f"Error detecting patterns for {ticker}: {str(e)}")
                results[ticker] = []
        
        return results


# Example usage and testing
if __name__ == "__main__":
    import yfinance as yf
    
    # Initialize recognizer
    recognizer = ChartPatternRecognizer()
    
    # Test with real stock data
    tickers = ["AAPL", "MSFT", "NVDA"]
    stock_data = {}
    
    print("="*60)
    print("Chart Pattern Recognition Test")
    print("="*60)
    
    for ticker in tickers:
        print(f"\nFetching data for {ticker}...")
        stock = yf.Ticker(ticker)
        df = stock.history(period="6mo")
        
        if not df.empty:
            stock_data[ticker] = df
            
            # Detect patterns
            patterns = recognizer.detect_patterns(df, confidence_threshold=0.3)
            
            print(f"\n{ticker} Pattern Detection Results:")
            if patterns:
                for pattern in patterns:
                    print(f"  {pattern.pattern_type}: {pattern.confidence:.1%} confidence")
                    if pattern.breakout_point:
                        print(f"    - Breakout detected at index {pattern.breakout_point}")
            else:
                print("  No patterns detected with sufficient confidence")
    
    # Test batch detection
    print("\n" + "="*60)
    print("Batch Pattern Detection")
    print("="*60)
    
    results = recognizer.batch_detect(stock_data, confidence_threshold=0.3)
    
    for ticker, patterns in results.items():
        print(f"\n{ticker}: {len(patterns)} patterns detected")
    
    print("\n" + "="*60)
    print(" Pattern recognition module ready!")
    print("Note: Model needs training data for accurate predictions")
    print("GPU acceleration: " + ("Enabled" if torch.cuda.is_available() else "Disabled"))