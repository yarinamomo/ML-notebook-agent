from minisweagent.agents.default import DefaultAgent
import src.utils.ui as ui

class UiAgent(DefaultAgent):
    def query(self) -> dict:
        with ui.wait_llm():
            response = super().query()
        actions = "\n".join([action.get("command", "") for action in response.get("extra", {}).get("actions", [])])
        ui.system(response.get("content", ""), actions=actions, step=self.n_calls)
        return response


    def execute_actions(self, message: dict) -> list[dict]:
        with ui.wait_tool(message.get("action")):
            response = super().execute_actions(message)
            for output in response:
                ui.agent(output.get("content", ""), action=output.get("action", None), return_code=output.get("returncode", None), step=self.n_calls)

        return response