"""Test the real upstream request boundary, with only the external model replaced."""
import json
import unittest
from smolagents.models import Model,ChatMessage,ChatMessageToolCall,ChatMessageToolCallFunction
from fallback_experiment import build_agent,TASK_CONSTRAINTS

class CapturingModel(Model):
    def __init__(self):super().__init__();self.requests=[]
    def generate(self,messages,**kwargs):
        self.requests.append([json.loads(m.model_dump_json()) for m in messages])
        index=len(self.requests)
        if index==3:return ChatMessage(role='assistant',content='{"amount_hkd":"727.66"}')
        name,args=('read_invoice',{'invoice_id':'INV-F01'}) if index==1 else ('get_fx_rate',{'currency':'USD'})
        return ChatMessage(role='assistant',content='',tool_calls=[ChatMessageToolCall(id=f'c{index}',type='function',function=ChatMessageToolCallFunction(name=name,arguments=args))])

class FallbackPromptTests(unittest.TestCase):
    def test_only_fallback_receives_extra_constraints_and_budget_stays_fixed(self):
        models=[];agents=[]
        for condition in ['original','preserved']:
            model=CapturingModel();agent=build_agent(model,[],condition)
            result=agent.run('Calculate the total settlement amount in HKD for invoice INV-F01.')
            self.assertEqual(result.state,'max_steps_error')
            self.assertEqual(agent.max_steps,2)
            self.assertEqual(len(model.requests),3)
            models.append(model);agents.append(agent)
        self.assertEqual(models[0].requests[:2],models[1].requests[:2])
        self.assertNotIn(TASK_CONSTRAINTS,models[0].requests[2][0]['content'][0]['text'])
        self.assertIn(TASK_CONSTRAINTS,models[1].requests[2][0]['content'][0]['text'])
        self.assertEqual(models[0].requests[2][1:],models[1].requests[2][1:])

    def test_prompt_configuration_does_not_leak_into_new_original_agent(self):
        patched=build_agent(CapturingModel(),[],'preserved')
        original=build_agent(CapturingModel(),[],'original')
        self.assertNotEqual(patched.prompt_templates['final_answer'],original.prompt_templates['final_answer'])
        self.assertNotIn(TASK_CONSTRAINTS,original.prompt_templates['final_answer']['pre_messages'])
        self.assertEqual(patched.prompt_templates['system_prompt'],original.prompt_templates['system_prompt'])
