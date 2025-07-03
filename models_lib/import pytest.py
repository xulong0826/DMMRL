import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
from models_lib.multi_modal import Multi_modal

import torch.nn as nn

class TestMultiModalForward:
    
    @pytest.fixture
    def mock_args(self):
        """Create mock arguments for Multi_modal initialization"""
        args = Mock()
        args.latent_dim = 128
        args.batch_size = 32
        args.graph = True
        args.sequence = True
        args.geometry = True
        args.seq_hidden_dim = 256
        args.gnn_hidden_dim = 256
        args.geo_hidden_dim = 256
        args.gnn_atom_dim = 74
        args.gnn_bond_dim = 14
        args.bias = True
        args.gnn_num_layers = 3
        args.dropout = 0.1
        args.gnn_activation = 'relu'
        args.seq_input_dim = 100
        args.seq_num_heads = 8
        args.seq_num_layers = 6
        args.vocab_num = 1000
        args.recons = True
        args.pro_num = 3
        args.task_type = 'class'
        args.pool_type = 'mean'
        args.fusion = 1
        args.norm = False
        args.output_dim = 1
        return args

    @pytest.fixture
    def mock_compound_config(self):
        """Create mock compound encoder config"""
        return {}

    @pytest.fixture
    def device(self):
        return torch.device('cpu')

    @pytest.fixture
    def sample_inputs(self, device):
        """Create sample inputs for forward method"""
        batch_size = 4
        seq_len = 50
        num_nodes = 20
        
        return {
            'trans_batch_seq': torch.randint(0, 1000, (batch_size, seq_len)),
            'seq_mask': torch.ones(batch_size * seq_len, dtype=torch.bool),
            'batch_mask_seq': torch.repeat_interleave(torch.arange(batch_size), seq_len),
            'gnn_batch_graph': torch.randn(num_nodes, 74),
            'gnn_feature_batch': torch.randn(num_nodes * 2, 14),
            'batch_mask_gnn': torch.repeat_interleave(torch.arange(batch_size), num_nodes // batch_size),
            'graph_dict': [Mock(), Mock()],
            'node_id_all': [torch.repeat_interleave(torch.arange(batch_size), num_nodes // batch_size)],
            'edge_id_all': torch.arange(num_nodes)
        }

    @patch('models_lib.multi_modal.MPNEncoder')
    @patch('models_lib.multi_modal.TrfmSeq2seq')
    @patch('models_lib.multi_modal.GeoGNNModel')
    def test_forward_all_modalities(self, mock_geo, mock_transformer, mock_gnn, 
                                   mock_args, mock_compound_config, device, sample_inputs):
        """Test forward method with all modalities enabled"""
        # Setup mocks
        mock_gnn_instance = Mock()
        mock_gnn_instance.return_value = torch.randn(20, 256)
        mock_gnn.return_value = mock_gnn_instance
        
        mock_transformer_instance = Mock()
        mock_transformer_instance.return_value = (torch.tensor(0.0), torch.randn(200, 256))
        mock_transformer.return_value = mock_transformer_instance
        
        mock_geo_instance = Mock()
        mock_geo_instance.return_value = (torch.randn(20, 256), torch.randn(10, 256))
        mock_geo.return_value = mock_geo_instance
        
        # Create model
        model = Multi_modal(mock_args, mock_compound_config, device)
        
        # Run forward
        cl_list, pred = model(**sample_inputs)
        
        # Assertions
        assert len(cl_list) == 3, "Should have 3 contrastive learning outputs for 3 modalities"
        assert pred.shape[0] == 4, "Batch size should be preserved"
        assert pred.shape[1] == mock_args.output_dim, "Output dimension should match args"
        
        # Verify all encoders were called
        mock_gnn_instance.assert_called_once()
        mock_transformer_instance.assert_called_once()
        mock_geo_instance.assert_called_once()

    @patch('models_lib.multi_modal.MPNEncoder')
    @patch('models_lib.multi_modal.TrfmSeq2seq')
    @patch('models_lib.multi_modal.GeoGNNModel')
    def test_forward_single_modality_graph_only(self, mock_geo, mock_transformer, mock_gnn,
                                               mock_args, mock_compound_config, device, sample_inputs):
        """Test forward with only graph modality"""
        mock_args.graph = True
        mock_args.sequence = False
        mock_args.geometry = False
        
        # Setup mocks
        mock_gnn_instance = Mock()
        mock_gnn_instance.return_value = torch.randn(20, 256)
        mock_gnn.return_value = mock_gnn_instance
        
        mock_transformer.return_value = Mock()
        mock_geo.return_value = Mock()
        
        model = Multi_modal(mock_args, mock_compound_config, device)
        cl_list, pred = model(**sample_inputs)
        
        assert len(cl_list) == 1, "Should have 1 contrastive learning output for graph only"
        assert pred.shape[0] == 4, "Batch size should be preserved"
        mock_gnn_instance.assert_called_once()

    @patch('models_lib.multi_modal.MPNEncoder')
    @patch('models_lib.multi_modal.TrfmSeq2seq')
    @patch('models_lib.multi_modal.GeoGNNModel')
    def test_forward_with_normalization(self, mock_geo, mock_transformer, mock_gnn,
                                       mock_args, mock_compound_config, device, sample_inputs):
        """Test forward with feature normalization enabled"""
        mock_args.norm = True
        
        # Setup mocks
        mock_gnn_instance = Mock()
        mock_gnn_instance.return_value = torch.randn(20, 256)
        mock_gnn.return_value = mock_gnn_instance
        
        mock_transformer_instance = Mock()
        mock_transformer_instance.return_value = (torch.tensor(0.0), torch.randn(200, 256))
        mock_transformer.return_value = mock_transformer_instance
        
        mock_geo_instance = Mock()
        mock_geo_instance.return_value = (torch.randn(20, 256), torch.randn(10, 256))
        mock_geo.return_value = mock_geo_instance
        
        model = Multi_modal(mock_args, mock_compound_config, device)
        cl_list, pred = model(**sample_inputs)
        
        assert len(cl_list) == 3
        assert pred.shape[0] == 4

    @patch('models_lib.multi_modal.MPNEncoder')
    @patch('models_lib.multi_modal.TrfmSeq2seq')
    @patch('models_lib.multi_modal.GeoGNNModel')
    def test_forward_fusion_strategies(self, mock_geo, mock_transformer, mock_gnn,
                                      mock_args, mock_compound_config, device, sample_inputs):
        """Test different fusion strategies"""
        # Setup mocks
        mock_gnn_instance = Mock()
        mock_gnn_instance.return_value = torch.randn(20, 256)
        mock_gnn.return_value = mock_gnn_instance
        
        mock_transformer_instance = Mock()
        mock_transformer_instance.return_value = (torch.tensor(0.0), torch.randn(200, 256))
        mock_transformer.return_value = mock_transformer_instance
        
        mock_geo_instance = Mock()
        mock_geo_instance.return_value = (torch.randn(20, 256), torch.randn(10, 256))
        mock_geo.return_value = mock_geo_instance
        
        # Test fusion strategy 1 (concatenation)
        mock_args.fusion = 1
        model1 = Multi_modal(mock_args, mock_compound_config, device)
        cl_list1, pred1 = model1(**sample_inputs)
        assert pred1.shape[0] == 4
        
        # Test fusion strategy 2 (element-wise multiplication)
        mock_args.fusion = 2
        model2 = Multi_modal(mock_args, mock_compound_config, device)
        cl_list2, pred2 = model2(**sample_inputs)
        assert pred2.shape[0] == 4
        
        # Test fusion strategy 0 (mean)
        mock_args.fusion = 0
        model0 = Multi_modal(mock_args, mock_compound_config, device)
        cl_list0, pred0 = model0(**sample_inputs)
        assert pred0.shape[0] == 1  # Mean fusion changes batch size

    @patch('models_lib.multi_modal.MPNEncoder')
    @patch('models_lib.multi_modal.TrfmSeq2seq')
    @patch('models_lib.multi_modal.GeoGNNModel')
    def test_forward_attention_pooling(self, mock_geo, mock_transformer, mock_gnn,
                                      mock_args, mock_compound_config, device, sample_inputs):
        """Test forward with attention-based pooling"""
        mock_args.pool_type = 'attention'
        
        # Setup mocks
        mock_gnn_instance = Mock()
        mock_gnn_instance.return_value = torch.randn(20, 256)
        mock_gnn.return_value = mock_gnn_instance
        
        mock_transformer_instance = Mock()
        mock_transformer_instance.return_value = (torch.tensor(0.0), torch.randn(200, 256))
        mock_transformer.return_value = mock_transformer_instance
        
        mock_geo_instance = Mock()
        mock_geo_instance.return_value = (torch.randn(20, 256), torch.randn(10, 256))
        mock_geo.return_value = mock_geo_instance
        
        model = Multi_modal(mock_args, mock_compound_config, device)
        cl_list, pred = model(**sample_inputs)
        
        assert len(cl_list) == 3
        assert pred.shape[0] == 4

    @patch('models_lib.multi_modal.MPNEncoder')
    @patch('models_lib.multi_modal.TrfmSeq2seq')
    @patch('models_lib.multi_modal.GeoGNNModel')
    def test_forward_shared_projectors(self, mock_geo, mock_transformer, mock_gnn,
                                      mock_args, mock_compound_config, device, sample_inputs):
        """Test forward with shared projectors (pro_num = 1)"""
        mock_args.pro_num = 1
        
        # Setup mocks
        mock_gnn_instance = Mock()
        mock_gnn_instance.return_value = torch.randn(20, 256)
        mock_gnn.return_value = mock_gnn_instance
        
        mock_transformer_instance = Mock()
        mock_transformer_instance.return_value = (torch.tensor(0.0), torch.randn(200, 256))
        mock_transformer.return_value = mock_transformer_instance
        
        mock_geo_instance = Mock()
        mock_geo_instance.return_value = (torch.randn(20, 256), torch.randn(10, 256))
        mock_geo.return_value = mock_geo_instance
        
        model = Multi_modal(mock_args, mock_compound_config, device)
        cl_list, pred = model(**sample_inputs)
        
        assert len(cl_list) == 3
        assert pred.shape[0] == 4

    @patch('models_lib.multi_modal.MPNEncoder')
    @patch('models_lib.multi_modal.TrfmSeq2seq')
    @patch('models_lib.multi_modal.GeoGNNModel')
    def test_forward_output_shapes(self, mock_geo, mock_transformer, mock_gnn,
                                  mock_args, mock_compound_config, device, sample_inputs):
        """Test that output shapes are correct for different configurations"""
        # Setup mocks
        mock_gnn_instance = Mock()
        mock_gnn_instance.return_value = torch.randn(20, 256)
        mock_gnn.return_value = mock_gnn_instance
        
        mock_transformer_instance = Mock()
        mock_transformer_instance.return_value = (torch.tensor(0.0), torch.randn(200, 256))
        mock_transformer.return_value = mock_transformer_instance
        
        mock_geo_instance = Mock()
        mock_geo_instance.return_value = (torch.randn(20, 256), torch.randn(10, 256))
        mock_geo.return_value = mock_geo_instance
        
        # Test different output dimensions
        for output_dim in [1, 10, 100]:
            mock_args.output_dim = output_dim
            model = Multi_modal(mock_args, mock_compound_config, device)
            cl_list, pred = model(**sample_inputs)
            
            assert pred.shape[1] == output_dim, f"Output dim should be {output_dim}"
            assert all(cl.shape[1] == mock_args.latent_dim for cl in cl_list), "CL outputs should match latent_dim"

    @patch('models_lib.multi_modal.MPNEncoder')
    @patch('models_lib.multi_modal.TrfmSeq2seq')
    @patch('models_lib.multi_modal.GeoGNNModel')
    def test_forward_device_consistency(self, mock_geo, mock_transformer, mock_gnn,
                                       mock_args, mock_compound_config, device, sample_inputs):
        """Test that all outputs are on the correct device"""
        # Setup mocks
        mock_gnn_instance = Mock()
        mock_gnn_instance.return_value = torch.randn(20, 256)
        mock_gnn.return_value = mock_gnn_instance
        
        mock_transformer_instance = Mock()
        mock_transformer_instance.return_value = (torch.tensor(0.0), torch.randn(200, 256))
        mock_transformer.return_value = mock_transformer_instance
        
        mock_geo_instance = Mock()
        mock_geo_instance.return_value = (torch.randn(20, 256), torch.randn(10, 256))
        mock_geo.return_value = mock_geo_instance
        
        model = Multi_modal(mock_args, mock_compound_config, device)
        cl_list, pred = model(**sample_inputs)
        
        assert pred.device == device, "Prediction should be on correct device"
        assert all(cl.device == device for cl in cl_list), "All CL outputs should be on correct device"

    @patch('models_lib.multi_modal.MPNEncoder')
    @patch('models_lib.multi_modal.TrfmSeq2seq')
    @patch('models_lib.multi_modal.GeoGNNModel')
    def test_forward_no_modalities(self, mock_geo, mock_transformer, mock_gnn,
                                  mock_args, mock_compound_config, device, sample_inputs):
        """Test forward when no modalities are enabled"""
        mock_args.graph = False
        mock_args.sequence = False
        mock_args.geometry = False
        
        mock_gnn.return_value = Mock()
        mock_transformer.return_value = Mock()
        mock_geo.return_value = Mock()
        
        model = Multi_modal(mock_args, mock_compound_config, device)
        
        # This should handle the empty x_list gracefully
        with pytest.raises((IndexError, RuntimeError)):
            cl_list, pred = model(**sample_inputs)