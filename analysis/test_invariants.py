"""Scientific invariants and data-construction regression checks."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from . import buffer_study as bs
from . import revision_study as rs
from .analyze import cvar

class StudyInvariants(unittest.TestCase):
    def test_auc_ties_matches_standard(self):
        score=np.array([.1,.1,.2,.5,.5,.5]);y=np.array([0,1,0,0,1,1],bool)
        self.assertAlmostEqual(bs.auc(score,y),roc_auc_score(y,score))
        self.assertEqual(bs.auc(np.ones(6),y),.5)

    def test_zero_probability_is_not_reported(self):
        s=rs.summarize(-1,np.full(1000,-1.0))
        self.assertAlmostEqual(s['p_boot'],2/1001)

    def test_undefined_draws_are_explicit(self):
        s=rs.summarize(-1,np.array([-2.,-1.,np.nan]))
        self.assertEqual(s['undefined_draws'],1)

    def test_sparse_returns_cannot_evade_risk_limit(self):
        d={'q':np.full(200,100.),'b':np.full(200,80.),'p':np.zeros(200),
           'r':np.arange(200)==0,'n':200}
        theta=bs.group_lp(d,np.zeros(200,dtype=int),1,2.,.99,True)
        self.assertAlmostEqual(theta[0],.04,places=7)
        self.assertAlmostEqual(cvar(bs.shortfall(d['q']*theta[0],d),.99),2.)

    def test_equal_value_benchmark_preserves_total_and_cap(self):
        b=np.array([20.,90.,50.]);q=np.array([100.,100.,100.])
        v=rs.scaled_base_equal(b,q,75.)
        self.assertAlmostEqual(v.mean(),75.)
        self.assertTrue(np.all(v<=q))

    def test_followup_and_proceeds_options(self):
        attrs={'contractResidualValue':'100','baseResidualValue':'80','vehicleValueAmount':'150',
               'acquisitionCost':'140','originalLeaseTermNumber':'36','vehicleTypeCode':'3',
               'vehicleManufacturerName':'TEST','terminationIndicator':'2','zeroBalanceCode':'1',
               'zeroBalanceEffectiveDate':'01/2024','scheduledTerminationDate':'01/2024'}
        rows=[dict(attrs,assetNumber='delayed',reportingPeriodEndDate='01-31-2024',liquidationProceedsAmount='0'),
              dict(attrs,assetNumber='delayed',reportingPeriodEndDate='04-30-2024',liquidationProceedsAmount='90'),
              dict(attrs,assetNumber='paid',reportingPeriodEndDate='01-31-2024',liquidationProceedsAmount='85'),
              dict(attrs,assetNumber='recent',zeroBalanceEffectiveDate='04/2024',reportingPeriodEndDate='04-30-2024',liquidationProceedsAmount='90'),
              dict(attrs,assetNumber='clock',terminationIndicator='',zeroBalanceCode='',reportingPeriodEndDate='06-30-2024',liquidationProceedsAmount='0')]
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'data.jsonl';p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
            primary=bs.load(p,'2024-01','2024-04')
            included=bs.load(p,'2024-01','2024-04',include_nonpositive=True)
            mature3=bs.load(p,'2024-01','2024-04',require_followup_months=6)
            mature6=bs.load(p,'2024-01','2024-04',proceeds_months=6,require_followup_months=6)
        self.assertEqual(set(primary['asset']),{'paid','recent'})
        self.assertEqual(set(included['asset']),{'delayed','paid','recent'})
        self.assertEqual(set(mature3['asset']),{'paid'})
        self.assertEqual(set(mature6['asset']),{'delayed','paid'})
        self.assertEqual(mature3['counts']['insufficient_followup'],1)

if __name__=='__main__':unittest.main()
