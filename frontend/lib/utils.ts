export function relativeTime(dateString: string): string {
  const date = new Date(dateString)
  const now = new Date()

  const diffMs = now.getTime() - date.getTime()

  const diffDays = Math.floor(
    diffMs / (1000 * 60 * 60 * 24)
  )

  if (diffDays === 0) return "Today"
  if (diffDays === 1) return "Yesterday"
  if (diffDays < 7) return `${diffDays} days ago`
  if (diffDays < 14) return "Last week"
  if (diffDays < 30)
    return `${Math.floor(diffDays / 7)} weeks ago`

  return `${Math.floor(diffDays / 30)} months ago`
}


