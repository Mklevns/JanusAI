"""
Tests for experiments/configs/run_gpt_2_attention_experiment.py
"""
import pytest
from unittest.mock import patch, MagicMock, ANY
import torch
import numpy as np

# Import classes to be tested
from experiments.configs.run_gpt_2_attention_experiment import (
    GPT2AttentionConfig,
    GPT2AttentionExperiment
)

# Mocks for external dependencies
MockGPT2Model = MagicMock(name="MockGPT2Model")
MockGPT2Tokenizer = MagicMock(name="MockGPT2Tokenizer")
MockEnhancedAIGrammar = MagicMock(name="MockEnhancedAIGrammar")
MockInterpretabilityReward = MagicMock(name="MockInterpretabilityReward")
MockFidelityCalculator = MagicMock(name="MockFidelityCalculator")
MockSymbolicDiscoveryEnv = MagicMock(name="MockSymbolicDiscoveryEnv")
MockPPOTrainer = MagicMock(name="MockPPOTrainer")
MockBaseExperiment = MagicMock(name="MockBaseExperiment") # If BaseExperiment has methods to mock

# --- Tests for GPT2AttentionConfig ---
class TestGPT2AttentionConfig:
    def test_default_initialization(self):
        config = GPT2AttentionConfig()
        assert config.model_name == 'gpt2'
        assert config.layer_index == 0
        assert config.head_index == 1
        assert config.max_complexity == 20
        assert config.num_text_samples == 100
        assert config.max_sequence_length == 64
        assert config.training_episodes == 200
        assert config.learning_rate == 3e-4
        assert config.gamma == 0.99
        assert config.reward_weights is not None # __post_init__ sets it
        assert config.reward_weights['fidelity'] == 0.6

    def test_custom_reward_weights(self):
        custom_weights = {'fidelity': 0.8, 'simplicity': 0.1, 'consistency': 0.05, 'insight': 0.05}
        config = GPT2AttentionConfig(reward_weights=custom_weights)
        assert config.reward_weights == custom_weights

# --- Tests for GPT2AttentionExperiment ---
# Patch all major external dependencies for the experiment class
@patch('experiments.configs.run_gpt_2_attention_experiment.GPT2Model', MockGPT2Model)
@patch('experiments.configs.run_gpt_2_attention_experiment.GPT2Tokenizer', MockGPT2Tokenizer)
@patch('experiments.configs.run_gpt_2_attention_experiment.EnhancedAIGrammar', MockEnhancedAIGrammar)
@patch('experiments.configs.run_gpt_2_attention_experiment.InterpretabilityReward', MockInterpretabilityReward)
@patch('experiments.configs.run_gpt_2_attention_experiment.FidelityCalculator', MockFidelityCalculator)
@patch('experiments.configs.run_gpt_2_attention_experiment.SymbolicDiscoveryEnv', MockSymbolicDiscoveryEnv)
@patch('experiments.configs.run_gpt_2_attention_experiment.PPOTrainer', MockPPOTrainer)
class TestGPT2AttentionExperiment:

    @pytest.fixture(autouse=True)
    def reset_global_mocks(self):
        MockGPT2Model.reset_mock()
        MockGPT2Tokenizer.reset_mock()
        MockEnhancedAIGrammar.reset_mock()
        MockInterpretabilityReward.reset_mock()
        MockFidelityCalculator.reset_mock()
        MockSymbolicDiscoveryEnv.reset_mock()
        MockPPOTrainer.reset_mock()
        # MockBaseExperiment might need reset if it's stateful or has class-level mocks

    @pytest.fixture
    def config(self):
        return GPT2AttentionConfig()

    @pytest.fixture
    def experiment(self, config):
        # Patch BaseExperiment.__init__ to avoid issues if it does complex things
        with patch.object(GPT2AttentionExperiment, '__bases__', (MockBaseExperiment,)):
             # MockBaseExperiment.__init__ = MagicMock() # If BaseExperiment init needs mocking
            exp = GPT2AttentionExperiment(config)
        return exp

    def test_experiment_init(self, experiment, config):
        assert experiment.config == config
        assert experiment.logger is not None
        # Other attributes should be None initially
        assert experiment.model is None
        assert experiment.tokenizer is None

    @patch.object(GPT2AttentionExperiment, '_load_model')
    @patch.object(GPT2AttentionExperiment, '_extract_attention_data')
    @patch.object(GPT2AttentionExperiment, '_create_variables')
    @patch.object(GPT2AttentionExperiment, '_setup_grammar')
    @patch.object(GPT2AttentionExperiment, '_setup_reward_calculator')
    @patch.object(GPT2AttentionExperiment, '_create_environment')
    @patch.object(GPT2AttentionExperiment, '_setup_trainer')
    def test_setup_calls_all_helpers(self, mock_setup_trainer, mock_create_env, mock_setup_reward,
                                     mock_setup_grammar, mock_create_vars, mock_extract_data,
                                     mock_load_model, experiment):
        experiment.setup()
        mock_load_model.assert_called_once()
        mock_extract_data.assert_called_once()
        mock_create_vars.assert_called_once()
        mock_setup_grammar.assert_called_once()
        mock_setup_reward.assert_called_once()
        mock_create_env.assert_called_once()
        mock_setup_trainer.assert_called_once()

    def test_load_model(self, experiment):
        mock_tokenizer_instance = MockGPT2Tokenizer.return_value
        mock_tokenizer_instance.pad_token = None # Simulate needing to set pad_token
        mock_tokenizer_instance.eos_token = "[EOS]"

        mock_model_instance = MockGPT2Model.return_value

        experiment._load_model()

        MockGPT2Tokenizer.from_pretrained.assert_called_once_with(experiment.config.model_name)
        assert mock_tokenizer_instance.pad_token == "[EOS]"
        MockGPT2Model.from_pretrained.assert_called_once_with(experiment.config.model_name, output_attentions=True)
        mock_model_instance.to.assert_called_once_with(ANY) # Check device
        mock_model_instance.eval.assert_called_once()
        assert experiment.model == mock_model_instance
        assert experiment.tokenizer == mock_tokenizer_instance

    @patch('experiments.configs.run_gpt_2_attention_experiment.GPT2AttentionExperiment._generate_text_samples')
    def test_extract_attention_data(self, mock_generate_samples, experiment, config):
        mock_generate_samples.return_value = ["sample text 1", "sample text 2"]

        # Mock tokenizer and model behavior
        mock_tokenizer_instance = MagicMock()
        experiment.tokenizer = mock_tokenizer_instance

        # Simulate tokenizer output
        tokenized_sample1 = {'input_ids': torch.tensor([[1,2,3]]), 'attention_mask': torch.tensor([[1,1,1]])}
        tokenized_sample2 = {'input_ids': torch.tensor([[4,5]]), 'attention_mask': torch.tensor([[1,1]])}
        mock_tokenizer_instance.side_effect = [tokenized_sample1, tokenized_sample2]
        mock_tokenizer_instance.convert_ids_to_tokens.side_effect = [["t1","t2","t3"], ["t4","t5"]]


        mock_model_instance = MagicMock()
        experiment.model = mock_model_instance

        # Simulate model output (attentions)
        # Batch=1, Heads=H (in GPT2Model output), SeqLen=S, SeqLen=S
        # Layer_index from config is 0. Head_index from config is 1.
        # Attention output for sample 1 (seq_len=3)
        attn_output1 = MagicMock()
        attn_tensor1 = torch.rand(1, 12, 3, 3) # GPT-2 base has 12 heads
        attn_output1.attentions = [attn_tensor1]
        # Attention output for sample 2 (seq_len=2)
        attn_output2 = MagicMock()
        attn_tensor2 = torch.rand(1, 12, 2, 2)
        attn_output2.attentions = [attn_tensor2]
        mock_model_instance.side_effect = [attn_output1, attn_output2]

        experiment._extract_attention_data()

        assert mock_generate_samples.call_count == 1
        assert mock_tokenizer_instance.call_count == 2 # For each sample text
        assert mock_model_instance.call_count == 2

        assert len(experiment.attention_data['attention_weights']) == 2
        assert experiment.attention_data['attention_weights'][0].shape == (3,3) # Head selected, batch dim removed
        assert experiment.attention_data['attention_weights'][1].shape == (2,2)
        assert experiment.attention_data['sequence_lengths'] == [3, 2]
        assert len(experiment.attention_data['token_types']) == 2
        # Example check for token types (depends on mock_tokenizer_instance.convert_ids_to_tokens)
        # If t1 starts with 'Ġ', type is 1, else 0.
        # Assuming all tokens here don't start with 'Ġ' for simplicity of this part.
        assert np.array_equal(experiment.attention_data['token_types'][0], np.array([0,0,0]))


    def test_generate_text_samples(self, experiment, config):
        config.num_text_samples = 5
        samples = experiment._generate_text_samples()
        assert len(samples) == 5
        assert samples[0] == "the cat sat on the mat." # Lowercase
        assert samples[1] == "A B A B A B A B"     # Uppercase

    @patch('JanusAI.experiments.configs.run_gpt_2_attention_experiment.Variable') # Patch Variable from its actual module
    def test_create_variables(self, MockDataclassVariable, experiment):
        experiment._create_variables()
        assert len(experiment.variables) == 6
        # Check that Variable constructor was called for each
        expected_calls = [
            ((('name', 'pos_diff'), ('index', 0)),), # Simplified tuple of tuples for args
            ((('name', 'pos_ratio'), ('index', 1)),),
            ((('name', 'token_type_i'), ('index', 2)),),
            ((('name', 'token_type_j'), ('index', 3)),),
            ((('name', 'relative_pos'), ('index', 4)),),
            ((('name', 'is_previous'), ('index', 5)),)
        ]
        # This is a bit verbose to check specific calls if order matters or if properties are complex.
        # For now, just check count of calls.
        assert MockDataclassVariable.call_count == 6
        # Example check for one variable properties if needed
        # call_args_list = MockDataclassVariable.call_args_list
        # assert call_args_list[0] == call(name='pos_diff', index=0, properties={})

    def test_setup_grammar(self, experiment):
        experiment._setup_grammar()
        MockEnhancedAIGrammar.assert_called_once()
        assert experiment.grammar == MockEnhancedAIGrammar.return_value
        experiment.grammar.add_custom_primitive.assert_called_once_with(
            'is_previous_token', ANY, 'pattern' # ANY for lambda
        )

    def test_setup_reward_calculator(self, experiment, config):
        experiment._setup_reward_calculator()
        MockInterpretabilityReward.assert_called_once_with(
            reward_weights=config.reward_weights,
            complexity_penalty_factor=0.01
        )
        assert experiment.reward_calculator == MockInterpretabilityReward.return_value
        assert experiment.reward_calculator.variables == experiment.variables # Check variables are assigned
        MockFidelityCalculator.assert_called_once()
        assert experiment.fidelity_calculator == MockFidelityCalculator.return_value

    def test_create_environment(self, experiment, config):
        # Need to have some data in self.attention_data and self.variables, self.grammar
        experiment.attention_data = {
            'attention_weights': [np.random.rand(2,2), np.random.rand(3,3)],
            'sequence_lengths': [2, 3],
            'token_types': [np.array([0,1]), np.array([1,0,0])]
        }
        experiment.variables = [MagicMock(), MagicMock()] # Dummy variables
        experiment.grammar = MockEnhancedAIGrammar.return_value
        experiment.reward_calculator = MockInterpretabilityReward.return_value

        experiment._create_environment()

        MockSymbolicDiscoveryEnv.assert_called_once()
        call_args = MockSymbolicDiscoveryEnv.call_args[1]
        assert call_args['grammar'] == experiment.grammar
        assert call_args['variables'] == experiment.variables
        assert call_args['reward_calculator'] == experiment.reward_calculator
        assert call_args['device'] == ANY # Device check
        assert 'inputs' in call_args['target_data']
        assert 'outputs' in call_args['target_data']
        # Total input rows = 2*2 + 3*3 = 4 + 9 = 13
        assert call_args['target_data']['inputs'].shape[0] == 13
        assert call_args['target_data']['outputs'].shape[0] == 13


    def test_setup_trainer(self, experiment, config):
        experiment.environment = MockSymbolicDiscoveryEnv.return_value # Needs env to be set
        experiment._setup_trainer()
        MockPPOTrainer.assert_called_once_with(
            env=experiment.environment,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            device=ANY # Device check
        )
        assert experiment.trainer == MockPPOTrainer.return_value

    def test_run(self, experiment):
        mock_trainer_instance = MockPPOTrainer.return_value
        mock_trainer_instance.train = MagicMock(return_value=[{'loss': 0.1}])
        mock_trainer_instance.get_best_solution = MagicMock(return_value=("best_expr_str", 0.9))
        experiment.trainer = mock_trainer_instance # Ensure trainer is set

        result = experiment.run()

        mock_trainer_instance.train.assert_called_once_with(num_episodes=experiment.config.training_episodes)
        mock_trainer_instance.get_best_solution.assert_called_once()
        assert result == "best_expr_str"
        assert experiment.training_history == [{'loss': 0.1}]

    @patch('torch.cuda.empty_cache')
    def test_teardown(self, mock_empty_cache, experiment):
        experiment.model = MockGPT2Model.return_value # Give it a model to delete
        experiment.teardown()
        # Check if model attribute is cleared (or if del was effective)
        # This is hard to directly assert `del` worked other than by side effects
        # or by checking if it's None if the code sets it to None.
        # The code `del self.model` removes the attribute.
        with pytest.raises(AttributeError): # Or check it's set to None if that's the impl
             _ = experiment.model

        if torch.cuda.is_available():
            mock_empty_cache.assert_called_once()
        else:
            mock_empty_cache.assert_not_called()


    def test_postprocess(self, experiment, config):
        sample_result = "discovered_expression_string"
        experiment.training_history = [{'step': 1, 'reward': 0.7}]

        processed_output = experiment.postprocess(sample_result)

        expected_output = {
            'discovered_expression': sample_result,
            'config': config.__dict__, # As per current implementation
            'training_history': experiment.training_history
        }
        assert processed_output == expected_output
