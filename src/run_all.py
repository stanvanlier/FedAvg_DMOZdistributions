def main():
    import src.data.make_dataset
    src.data.make_dataset.main()

    import src.features.build_features
    src.features.build_features.main()

    import src.models.multiprocessing
    src.models.multiprocessing.run()


if __name__ == "__main__":
    main()
