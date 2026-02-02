import mini_swe_agent_examples as run_mini_agent


def main():
    target_nb_instance = "sklearn_1"
    source_path = f"example/JunoBench/{target_nb_instance}"
    run_mini_agent.example_with_agent(source_path, problem_mode="JunoBench_Buggy")


if __name__ == "__main__":
    main()
