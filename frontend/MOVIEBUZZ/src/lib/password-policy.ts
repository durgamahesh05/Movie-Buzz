export const PASSWORD_POLICY_MESSAGE =
  "Password must be at least 6 characters and include 1 uppercase letter, 1 number, and 1 special character.";

export function getPasswordPolicyError(password: string) {
  if (password.length < 6) {
    return PASSWORD_POLICY_MESSAGE;
  }
  if (!/[A-Z]/.test(password)) {
    return PASSWORD_POLICY_MESSAGE;
  }
  if (!/\d/.test(password)) {
    return PASSWORD_POLICY_MESSAGE;
  }
  if (!/[^A-Za-z0-9\s]/.test(password)) {
    return PASSWORD_POLICY_MESSAGE;
  }
  return "";
}
