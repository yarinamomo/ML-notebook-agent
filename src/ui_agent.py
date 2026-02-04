from minisweagent.agents.default import DefaultAgent
import src.utils.ui as ui

class UiAgent(DefaultAgent):
    def query(self) -> dict:
        with ui.wait_llm():
            response = super().query()
        ui.system(response.get("content", ""), step=self.model.n_calls)
        return response


    def get_observation(self, response: dict) -> dict:
        response = super().get_observation(response)
        ui.agent(response.get("output", ""), action=response.get("action", None), return_code=response.get("returncode", None), step=self.model.n_calls)
        return response


    def execute_action(self, action: dict) -> dict:
        with ui.wait_tool(action.get("action")):
            return super().execute_action(action)