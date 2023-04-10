module.exports = {
  root: true,
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module'
  },
  extends: ['eslint:recommended', 'prettier'],
  env: {
    es2021: true,
    node: true
  },
  rules: {
    'no-console': 'off',
    'no-unused-vars': 'off',
    'no-async-promise-executor': 'off',
    'require-yield': 'off'
  }
};
