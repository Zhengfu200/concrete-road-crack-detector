"""Guard the reproducibility boundary of the frozen video entry."""
import copy
import pytest

from common import digest,write_json
from preprocessing import preprocessing_spec
import run_frozen_video as frozen

@pytest.fixture
def contract(tmp_path,monkeypatch):
    weights=tmp_path/'model.pt';weights.write_bytes(b'fixture checkpoint')
    threshold=tmp_path/'thresholds.json'
    write_json(weights.with_suffix('.release.json'),{'provenance_sha256':'fixture-data'})
    write_json(threshold,{'thresholds':{'0':.18819282948970795},'weights_sha256':digest(weights),
        'provenance_sha256':'fixture-data','imgsz':960,'preprocessing':preprocessing_spec(960,.3)})
    monkeypatch.setattr(frozen,'resolve',lambda name:tmp_path/name)
    return {'weights':'model.pt','thresholds':'thresholds.json','confidence':.18819282948970795,
        'imgsz':960,'nms_iou':.3,'files':{'model.pt':digest(weights),'thresholds.json':digest(threshold)},'packages':{}}

def test_changed_checkpoint_is_rejected_before_inference(contract,tmp_path):
    (tmp_path/'model.pt').write_bytes(b'other checkpoint')
    with pytest.raises(ValueError,match='冻结文件已改变'):
        frozen.validate_contract(contract)

def test_threshold_and_preprocessing_cannot_drift(contract,tmp_path):
    weights,thresholds=frozen.validate_contract(contract)
    assert weights.name=='model.pt' and thresholds.name=='thresholds.json'
    wrong=copy.deepcopy(contract);wrong['confidence']=.5
    with pytest.raises(ValueError,match='不匹配'):frozen.validate_contract(wrong)
    wrong=copy.deepcopy(contract);wrong['nms_iou']=.5
    with pytest.raises(ValueError,match='预处理版本不匹配'):frozen.validate_contract(wrong)

def test_environment_change_is_rejected(contract,monkeypatch):
    contract['packages']={'ultralytics':'old-version'}
    monkeypatch.setattr(frozen,'version',lambda name:'new-version')
    with pytest.raises(ValueError,match='运行环境已变化'):frozen.validate_contract(contract)
