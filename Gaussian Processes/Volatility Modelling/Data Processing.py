import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from datetime import timedelta
from dotenv import load_dotenv
from client import FtxClient
from dataclasses import dataclass
from functools import partial

class OrderbookMixin:               
    """
    Calculates order book features from transactions level data 
    """
    def get_depth(self, snap, midprice, level):
        subset = snap[np.abs(snap['price']-midprice)/midprice < level]
        return subset['size'].sum()
    
    def process_snapshot(self, snap):
        N = 5 # N*2 := number of orderbook bins stradding price
        levels = np.array(list(range(1,N+1)))/N 
        
        # subsetting orders
        t = snap['time'].min()
        bid = snap.query("side == 'buy'")
        ask = snap.query("side == 'sell'")
        
        if len(bid) > 0 and len(ask) > 0:
            # best bid and ask + spread
            bb = bid[bid['price'] == bid['price'].max()].iloc[0]
            ba = ask[ask['price'] == ask['price'].min()].iloc[0]
            spread = bb['price']-ba['price']


            # price calculations
            vwap = np.sum(snap['price']*snap['size'])/snap['size'].sum()
            midprice = (bb['price']*bb['size'] +  ba['price']*ba['size'])/(bb['size']+ba['size'])

            # orderbook
            bid_depth = {f'bid-{l}': self.get_depth(bid[['price','size']], midprice, l) for l in levels}
            ask_depth = {f'ask-{l}': self.get_depth(ask[['price','size']], midprice, l) for l in levels}

            return {'time':t, 'midprice':midprice, 'vwap':vwap, 'spread':spread, **bid_depth, **ask_depth}
        else:
            print('Dropped Dates:',t)
            # orderbook
            bid_depth = {f'bid-{l}': np.nan for l in levels}
            ask_depth = {f'ask-{l}': np.nan for l in levels}
            return {'time':np.nan,'midprice':np.nan,'vwap':np.nan,'spread':np.nan,**bid_depth,**ask_depth}

@dataclass
class FTXData(OrderbookMixin):
    client : FtxClient
    markets : list
    date_range : list
    time_scale : str
    
    def __post_init__(self):
        self.date_range = list(map(self._convert_dates, self.date_range))
        self.dt = timedelta(minutes = self.time_scale)
        self.raw_data = {m:self._get_raw_trades(m) for m in self.markets}
        self.data = self._process_trades()
        
    def __getitem__(self, market):
        assert market in self.markets
        return self.data[market]
    
    def _convert_dates(self, date):
        return pd.to_datetime(date, format = "%Y-%m-%d").timestamp()
    
    def _get_raw_trades(self, market):
        """
        Returns DataFrame with raw transaction level data from FTX 
        """
        print('Querying raw data...')
        trades = self.client.get_all_trades(market, *self.date_range)
        df = pd.DataFrame(trades).sort_values(by = 'time')
        df.reset_index(inplace = True, drop = True)
        df.loc[:,'time'] = pd.to_datetime(df['time'])
        return df
    
    def _get_snapshots(self, trades):
        print('Collecting Snapshots...')
        i, snapshots = 0, []
        t0, t1 = trades.time.agg(['min','max'])
        while t0+self.dt*(i+1) < t1:
            snap = trades[(trades['time'] > t0+self.dt*i) & (trades['time'] < t0+self.dt*(i+1))] 
            snapshots.append(snap)
            i+=1    
        return snapshots
    
    def _process_trades(self):
        """
        Creating orderbook features
        """
        print('Processing Trades...')
        trade_dict = {}
        for m in self.markets:
            snaps = self._get_snapshots(self.raw_data[m])
            clean_snaps = list(map(self.process_snapshot, snaps)) # parallelize later
            trade_dict[m] = pd.DataFrame(clean_snaps).dropna()
        print('Complete.')
        return trade_dict
    
    def save(self, path, file_name):
        """
        Returns a folder with a csv for each market
        """
        folder = os.path.join(path, file_name)
        os.mkdir(folder)
        for m in self.markets:
            df = self.data[m]
            f_name = os.path.join(folder, m.replace('/','-'))
            df.to_csv(f_name+'.csv', index = False, encoding = 'utf-8')

@dataclass
class Dataset:
    
    _data_ : FTXData # Processed orderbook data
    price_type : str # returns defined by midprice of orderbook or volume weighted returns
    
    def __post_init__(self):
        self.markets = self._data_.markets
        self.envelopes = self._process_order_book()
                
    def __getitem__(self, market):
        assert market in self._data_.markets
        return self._data_[market]
    
    @property
    def OB_columns(self):
        return [c for c in self._data_[self.markets[0]].columns if '-' in c]
    
    def _process_order_book(self):
        """
        Returns dictionary with envelopes for two data types
        
        Note that
            - Pure: Original return sequence
            - Split:  Positive return sequence and negative return sequence
        """
        # initilizing parent dictionary
        data_dict = {}
        
        for m in self._data_.markets:
            # calculating returns and dropping unecessary columns for modelling
            prices = self._data_[m][self.price_type]
            self._data_[m]['log_ret'] = np.log(prices) - np.log(prices.shift(1))
            self._data_[m]['t'] = list(range(len(prices)))
            self._data_[m].reset_index(drop = True, inplace = True)
            
            # envelope of pure returns 
            # (can completely remove this later - only used for research/motivation)
            env_dict = {}
            UL = ['upper', 'lower']
            envelope = [self._get_envelope(self._data_[m], x) for x in UL]
            env_dict['pure'] = dict(zip(UL, envelope))  
            
            # splitting returns into positive and negative return sequence
            # creating envelope characterized by superposition of both
            pos = self._data_[m].query('log_ret >= 0')
            neg = self._data_[m].query('log_ret < 0')
            env_dict['split'] = {
                'pos': pos[['t', 'log_ret']], 
                'neg': neg[['t', 'log_ret']],
                'upper': self._get_envelope(pos, 'upper'),
                'lower': self._get_envelope(neg, 'lower')}
            data_dict[m] = env_dict
        return data_dict
                           
    def _get_envelope(self, df, env_type):
        """
        Calculating envelope of log_ret and summing orderbooks 
        of observations pairs for a single asset
        """
        rows = [] 
        OB_cols = self.OB_columns # orderbook col names
        
        ## _TEMP_
        U_idx, L_idx = [0], [0]
        for i in range(1, len(df)-1):
            r0, r1, r2 = df['log_ret'].iloc[[i-1,i,i+1]]
            if r1 >= r0 and r1 >= r2 and env_type == 'upper':
                series = df[OB_cols].iloc[U_idx[-1]+1:i+1].sum() 
                series['log_ret'] = r1
                series['t'] = df['t'].iloc[i]
                rows.append(series)
                U_idx.append(i)
            elif r1 <= r0 and r1 <= r2 and env_type == 'lower':
                series = df[OB_cols].iloc[L_idx[-1]+1:i+1].sum()
                series['log_ret'] = r1
                series['t'] = df['t'].iloc[i]
                rows.append(series)
                L_idx.append(i)
        return pd.DataFrame(rows)