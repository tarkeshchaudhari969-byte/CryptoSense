import unittest
import numpy as np
import pandas as pd
from app.investor import risk_lab

class InvestorRiskTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(7)
        dates=pd.date_range('2021-01-01', periods=500, freq='D')
        base=rng.normal(.0005,.025,len(dates))
        self.frames={}
        for sym,scale in [('BTC',1),('ETH',1.25),('SOL',1.6)]:
            r=base*0.5+rng.normal(0,.018*scale,len(dates))
            close=100*np.exp(np.cumsum(r))
            self.frames[sym]=pd.DataFrame({'date':dates,'close':close})

    def test_risk_lab_returns_expected_sections(self):
        out=risk_lab([{'symbol':'BTC','quantity':.5},{'symbol':'ETH','quantity':2}],self.frames)
        self.assertGreater(out['portfolio']['value'],0)
        self.assertGreaterEqual(out['risk']['observations'],400)
        self.assertEqual(len(out['scenarios']),4)
        self.assertGreater(out['costs']['estimated_round_trip_cost'],0)
        self.assertGreater(out['position_sizing']['max_position_notional'],0)

    def test_invalid_duplicate_and_quantity_rejected(self):
        with self.assertRaises(ValueError): risk_lab([{'symbol':'BTC','quantity':1},{'symbol':'BTC','quantity':2}],self.frames)
        with self.assertRaises(ValueError): risk_lab([{'symbol':'BTC','quantity':-1}],self.frames)

    def test_single_asset_is_supported(self):
        out=risk_lab([{'symbol':'SOL','quantity':1}],self.frames)
        self.assertEqual(out['risk']['correlation']['labels'],['SOL'])
        self.assertEqual(out['risk']['diversification_score'],0)

if __name__=='__main__': unittest.main()
