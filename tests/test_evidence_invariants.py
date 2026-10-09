import unittest, copy, random
import evidence_invariants as audit
from sim_common import RuleProxy

class EvidenceTests(unittest.TestCase):
 def fixture(self):
  items={'P001':{'role':'problem','vector':[1.,0.]},'P002':{'role':'problem','vector':[0.,1.]},'S001':{'role':'solution','vector':[0.,1.]},'S002':{'role':'solution','vector':[1.,0.]}}
  d={'run_manifest':{'n_ticks':1},'solution_decoupling':{'S001':'S001','S002':'S002'},'arrival_schedule':dict.fromkeys(items,0),'condition':'autonomous','couplings':[{'tick':0,'problem_id':'P001','solution_id':'S002','cosine_similarity':1.}]}
  events=[{'tick':0,'problem_id':'P001','repeat':i,'parsed':{'P001':'S002'}} for i in range(3)]
  return d,items,events
 def test_valid(self):
  d,i,e=self.fixture(); self.assertEqual(audit.validate_cell(d,i,e)['failures'],[])
 def test_wrong_similarity(self):
  d,i,e=self.fixture();d['couplings'][0]['cosine_similarity']=.7;self.assertTrue(audit.validate_cell(d,i,e)['failures'])
 def test_origin(self):
  d,i,e=self.fixture();d['couplings'][0]['solution_id']='S001';self.assertTrue(audit.validate_cell(d,i,e)['failures'])
 def test_arrival(self):
  d,i,e=self.fixture();d['arrival_schedule']['S002']=1;self.assertTrue(audit.validate_cell(d,i,e)['failures'])
 def test_duplicate(self):
  d,i,e=self.fixture();d['couplings']*=2;self.assertTrue(audit.validate_cell(d,i,e)['failures'])
 def test_replay_disagrees(self):
  d,i,e=self.fixture();e[0]['parsed']['P001']=None;e[1]['parsed']['P001']=None;self.assertTrue(audit.validate_cell(d,i,e)['failures'])
 def test_missing_repeat(self):
  d,i,e=self.fixture();self.assertTrue(audit.validate_cell(d,i,e[:2])['failures'])
 def test_proxy_energy_and_gate(self):
  p=RuleProxy(random.Random(11));
  for _ in range(6):
   self.assertTrue(p.can_evaluate('P'));self.assertFalse(p.evaluate('P',.59)['accepted'])
  self.assertFalse(p.can_evaluate('P'))
  self.assertTrue(p.evaluate('Q',.60)['accepted'])
 def test_proxy_noise_disabled(self):
  p=RuleProxy(random.Random(11))
  for _ in range(6):self.assertFalse(p.evaluate('P',.59)['fidelity_degraded'])
 def test_invalid_minority_is_reported_not_committed(self):
  d,i,e=self.fixture();e[2]['parsed']['P001']='S001';r=audit.validate_cell(d,i,e)
  self.assertEqual(r['failures'],[]);self.assertEqual(len(r['response_anomalies']),1)
 def test_invalid_mode_fails(self):
  d,i,e=self.fixture()
  for x in e:x['parsed']['P001']='S001'
  self.assertTrue(audit.validate_cell(d,i,e)['failures'])
 def test_production_l1_on_synthetic_inputs(self):
  from unittest.mock import patch
  import run_layer1_human as production
  d,items,e=self.fixture()
  payload={'items':items,'embedding_model':{'model_id':'synthetic','version_pinned':'test'}}
  captured={}
  def save(seed,result):captured.update(result);return 'in-memory-test'
  with patch.object(production,'load_embeddings',return_value=payload), patch.object(production,'write_layer1_result',side_effect=save):
   production.run(seed=11,corpus_id='synthetic',embedding_model_key='synthetic',n_ticks=3)
  self.assertEqual(audit.validate_cell(captured,items)['failures'],[])
  self.assertGreater(len(captured['evaluations']),0)
