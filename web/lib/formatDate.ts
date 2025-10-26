export const formatDate = (dateStr: string) => {
  // Parse the date and force it to be interpreted at noon UTC
  // This ensures the date doesn't shift due to timezone differences
  const [year, month, day] = dateStr.split("-").map((num) => parseInt(num, 10));
  const date = new Date(Date.UTC(year, month - 1, day, 12, 0, 0));

  return date.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    timeZone: "UTC", // Ensure consistent date display regardless of local timezone
  });
};
