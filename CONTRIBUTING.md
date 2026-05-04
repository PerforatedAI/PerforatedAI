# Contributing Guide

Welcome! We are glad that you want to contribute to Perforated AI!

* [Ways to Contribute](#ways-to-contribute)
* [Find an Issue](#find-an-issue)
* [Ask for Help](#ask-for-help)
* [Development Environment Setup](#development-environment-setup)
* [Pull Request Process](#pull-request-process)
* [Sign Your Commits](#sign-your-commits)
* [Pull Request Checklist](#pull-request-checklist)

As you get started, you are in the best position to give us feedback on areas of our project that we need help with including:

* Problems found during setting up a new developer environment
* Gaps in our documentation or examples
* Bugs in our code or automation

If anything doesn't make sense, or doesn't work when you run it, please open a bug report and let us know!

## Ways to Contribute

We welcome many different types of contributions including:

* New features
* Bug fixes
* Documentation improvements
* Example code and tutorials
* Issue triage
* Performance improvements
* Testing improvements
* Code reviews

Not everything happens through a GitHub pull request. Feel free to open an issue to discuss your ideas before implementing them.

## Find an Issue

We label issues to help new contributors find good starting points:

* **good first issue**: Extra information to help you make your first contribution
* **help wanted**: Issues suitable for any contributor

If you want to contribute but can't find a suitable issue, feel free to open an issue asking for suggestions.

Once you find an issue you'd like to work on, please post a comment saying you want to work on it to avoid duplicate effort.

## Ask for Help

The best way to reach us with questions is to:

* Comment on the original GitHub issue
* Open a new issue with your question
* Check the documentation in the `API/` and `Examples/` directories

## Development Environment Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/PerforatedAI.git
   cd PerforatedAI
   ```

2. **Set up Python environment**
   ```bash
   python -m venv ENV
   source ENV/bin/activate  # On Windows: ENV\Scripts\activate
   ```

3. **Install PerforatedAI in development mode**
   ```bash
   pip install -e .
   ```

See the [README.md](README.md) for additional setup instructions.

## Pull Request Process

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** with clear, descriptive commit messages
3. **Test your changes** thoroughly
4. **Update documentation** if you're changing functionality
5. **Submit a pull request** with a clear description of the changes

## Sign Your Commits

### Developer Certificate of Origin (DCO)

Licensing is important to open source projects. We require that contributors sign off on commits submitted to our project's repositories. The [Developer Certificate of Origin (DCO)](https://developercertificate.org/) certifies that you wrote and have the right to contribute the code you are submitting to the project.

You sign-off by adding the following to your commit messages:

```
This is my commit message

Signed-off-by: Your Name <your.name@example.com>
```

Git has a `-s` command line option to do this automatically:

```bash
git commit -s -m 'This is my commit message'
```

If you forgot to do this and have not yet pushed your changes to the remote repository, you can amend your commit with the sign-off by running:

```bash
git commit --amend -s
```

## Pull Request Checklist

When you submit your pull request, please check the following:

* [ ] Code follows the existing style and conventions
* [ ] Changes are well-documented (docstrings, comments, README updates)
* [ ] All tests pass (if tests exist)
* [ ] Commit messages are clear and descriptive
* [ ] Commits are signed off (DCO)
* [ ] PR description explains what changes were made and why

---

Thank you for contributing to Perforated AI! Your efforts help make this project better for everyone. 🎉
