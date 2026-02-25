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
        actions = message.get("extra", {}).get("actions", []) # command, tool_call_id
        commands = "\n".join([action.get("command", "") for action in actions])
        with ui.wait_tool(commands):
            response_list = super().execute_actions(message)
            for response in response_list:
                tool_call_id = response.get('tool_call_id', '')
                for action in actions:
                    if action.get('tool_call_id', '') == tool_call_id:
                        cmd = action.get('command', '')
                        break
                extra = response.get('extra', {})
                output = extra.get('raw_output', '')
                returncode = extra.get('returncode', '')
                # timestamp = extra.get('timestamp', '')
                # exception_info = extra.get('exception_info', '')
                ui.agent(output, action=cmd, return_code=returncode, step=self.n_calls)

        return response_list