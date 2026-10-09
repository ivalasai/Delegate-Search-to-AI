import unittest,random
import generate_identity_exports as g

class IdentityTests(unittest.TestCase):
    def payload(self,mapping,sid):
        return {'solution_decoupling':mapping,'couplings':[{'problem_id':'P001','solution_id':sid}]}
    def test_same_content_different_labels(self):
        a=self.payload({'S001':'S010','S002':'S020'},'S001')
        b=self.payload({'S001':'S020','S002':'S010'},'S002')
        self.assertEqual(g.pair_set(a),g.pair_set(b))
    def test_same_label_different_content(self):
        self.assertNotEqual(g.pair_set(self.payload({'S001':'S010'},'S001')),g.pair_set(self.payload({'S001':'S020'},'S001')))
    def test_missing_mapping_rejected(self):
        with self.assertRaises(ValueError): g.pair_set({'couplings':[]})
    def test_nonbijective_mapping_rejected(self):
        with self.assertRaises(ValueError): g.pair_set(self.payload({'S001':'S010','S002':'S010'},'S001'))
    def test_unknown_id_rejected(self):
        with self.assertRaises(ValueError): g.pair_set(self.payload({'S001':'S010'},'S002'))
    def test_null_distinct_pools(self):
        a={('P001','S001')}; b={('P002','S002')}
        x=g.null_comparison([a],a,b,random.Random(7),right_pool=[b])
        self.assertEqual(x['null_mean'],0)
        self.assertEqual(x,g.null_comparison([a],a,b,random.Random(7),right_pool=[b]))
    def test_null_bad_size_rejected(self):
        with self.assertRaises(ValueError):g.null_comparison([set()],{('P001','S001')},{('P001','S001')},random.Random(7))
    def test_main_analysis_uses_same_identity(self):
        from analyze_variance import coupling_pairs
        p=self.payload({'S001':'S010'},'S001')
        self.assertEqual(coupling_pairs(p),{('P001','S010')})
    def test_top_uses_content(self):
        rows=[{'model':'A','seed':11,'top_solutions':[('S001',4)],'top_solutions_origin':[('S010',4)]},{'model':'A','seed':22,'top_solutions':[('S002',4)],'top_solutions_origin':[('S010',4)]}]
        self.assertEqual(g.top_jaccards(rows,'A')['11_vs_22'],1)

if __name__=='__main__':unittest.main()
